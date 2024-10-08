import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from PEReplay import PrioritizedExperienceReplay

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN:
    def __init__(self, input_dim, output_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)
        self.memory = PrioritizedExperienceReplay(10000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.priority_epsilon = 0.01
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_beta_increment = 0.001

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(3)]], device=self.device, dtype=torch.long)

    def store_transition(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        max_priority = self.memory.max_priority()
        self.memory.add(max_priority, transition)

    def optimize_model(self):
        if self.memory.n_entries < self.batch_size:
            return 0, 0

        batch, idxs, is_weights = self.memory.sample(self.batch_size, self.priority_beta)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        state_batch = torch.cat(state_batch).to(self.device)
        action_batch = torch.cat(action_batch).to(self.device)
        reward_batch = torch.cat(reward_batch).to(self.device)
        next_state_batch = torch.cat(next_state_batch).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = (q_values - expected_q_values.unsqueeze(1)).pow(2) * torch.tensor(is_weights, device=self.device).unsqueeze(1)
        prios = loss + self.priority_epsilon
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        for idx, prio in zip(idxs, prios.detach().cpu().numpy()):
            self.memory.update(idx, prio[0])

        return loss.item(), q_values.mean().item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_scheduler(self):
        self.scheduler.step()
        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
