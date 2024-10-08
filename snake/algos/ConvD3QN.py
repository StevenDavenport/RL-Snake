import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from PEReplay import PrioritizedExperienceReplay

class ConvD3QN(nn.Module):
    def __init__(self, input_shape, n_actions, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=32):
        super(ConvD3QN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = PrioritizedExperienceReplay(memory_size)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the output from the last conv layer
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2])))
        linear_input_size = convw * convh * 64

        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)

        # Target network
        self.target_net = None

        # PER parameters
        self.priority_epsilon = 0.01
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_beta_increment = 0.001

        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten

        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals

    def init_target_net(self):
        self.target_net = ConvD3QN(self.input_shape, self.n_actions).to(self.device)
        target_state_dict = self.state_dict()
        # Remove the 'target_net.' prefix from the keys
        target_state_dict = {k.replace('target_net.', ''): v for k, v in target_state_dict.items() if not k.startswith('target_net.')}
        self.target_net.load_state_dict(target_state_dict)
        self.target_net.eval()

    def select_action(self, state):
        random_number = random.random()
        if random_number > self.epsilon:
            with torch.no_grad():
                action = self(state).max(1)[1].view(1, 1)
                print(f"CHOSE GREEDY ACTION: {action.item()}")
                return action
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long, device=self.device)
            print(f"CHOSE RANDOM ACTION: {action.item()}")
            return action

    def store_transition(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        max_priority = self.memory.max_priority()
        self.memory.add(max_priority, transition)

    def optimize_model(self):
        if self.memory.n_entries < self.batch_size:
            return 0, 0

        batch, idxs, is_weights = self.memory.sample(self.batch_size, self.priority_beta)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.cat(state_batch).to(self.device)
        action_batch = torch.cat(action_batch).to(self.device)
        reward_batch = torch.cat(reward_batch).to(self.device)
        next_state_batch = torch.cat(next_state_batch).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float).to(self.device)

        # Double DQN
        next_state_values = self.target_net(next_state_batch).detach()
        next_action = self(next_state_batch).max(1)[1].unsqueeze(1)
        next_state_values = next_state_values.gather(1, next_action).squeeze(1)

        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))

        state_action_values = self(state_batch).gather(1, action_batch)

        loss = (state_action_values - expected_state_action_values.unsqueeze(1)).pow(2) * torch.tensor(is_weights, device=self.device).unsqueeze(1)
        prios = loss + self.priority_epsilon
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        for idx, prio in zip(idxs, prios.detach().cpu().numpy()):
            self.memory.update(idx, prio[0])

        return loss.item(), state_action_values.mean().item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        print(f"Updated epsilon: {self.epsilon:.4f}")

    def update_scheduler(self):
        self.scheduler.step()
        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()