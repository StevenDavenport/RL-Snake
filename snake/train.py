import random
import torch
from snake_env import SnakeEnv
from algos.DQN import DQN
from algos.ConvDQN import ConvDQN
from algos.ConvD3QN import ConvD3QN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def train(max_episodes=10000, checkpoint_interval=500, model_type='ConvD3QN'):
    env = SnakeEnv(render_mode=False)  # No rendering during training
    input_shape = env.observation_space.shape  # (4, n_rows, n_cols)
    print(f"Input shape: {input_shape}")
    n_actions = env.action_space.n

    if model_type == 'DQN':
        model = DQN(input_shape, n_actions)
        print("DQN model initialized")
    elif model_type == 'ConvDQN':
        model = ConvDQN(input_shape, n_actions)
        print("ConvDQN model initialized")
    elif model_type == 'ConvD3QN':
        model = ConvD3QN(input_shape, n_actions)
        print("ConvD3QN model initialized")
    else:
        raise ValueError("Invalid model type. Choose 'DQN', 'ConvDQN', or 'ConvD3QN'.")

    model.init_target_net()

    # Set up the plots
    plt.ion()  # Turn on interactive mode
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    rewards_plot, = ax1.plot([], [], 'b-', alpha=0.3)
    rewards_avg_plot, = ax1.plot([], [], 'r-')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    
    lengths_plot, = ax2.plot([], [], 'g-', alpha=0.3)
    lengths_avg_plot, = ax2.plot([], [], 'r-')
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    
    losses_plot, = ax3.plot([], [], 'm-', alpha=0.3)
    losses_avg_plot, = ax3.plot([], [], 'r-')
    ax3.set_title('Losses')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    
    q_values_plot, = ax4.plot([], [], 'c-', alpha=0.3)
    q_values_avg_plot, = ax4.plot([], [], 'r-')
    ax4.set_title('Average Q-values')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Q-value')

    plt.tight_layout()

    episode_rewards = []
    episode_lengths = []
    losses = []
    avg_q_values = []
    longest_episode = 0

    patience = 1000  # Number of episodes to wait for improvement
    best_reward = float('-inf')
    episodes_without_improvement = 0

    for episode in range(max_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(model.device)
        episode_reward = 0
        done = False
        step = 0
        episode_loss = 0
        episode_q_value = 0
        snake_length = 1  # Initialize snake length

        print(f"\nEpisode {episode + 1}")

        while not done:
            action = model.select_action(state)
            next_state, reward, done, info = env.step(action.item())
            
            snake_length = info.get('snake_length', 1)  # Default to 1 if 'snake_length' is not in info

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(model.device)
            reward = torch.tensor([reward], device=model.device)
            done = torch.tensor([float(done)], device=model.device)

            model.store_transition(state, action, reward, next_state, done)
            state = next_state

            loss, avg_q = model.optimize_model()
            episode_loss += loss
            episode_q_value += avg_q
            episode_reward += reward.item()
            step += 1

            print(f"Action: {action.item()}, Reward: {reward.item()}, Done: {done.item()}")
            print(f"Epsilon: {model.epsilon:.4f}")
            print(f"Random number: {random.random():.4f}")
            print(f"Snake Length: {snake_length}")
            print(f"Q-values: {model(state).detach().cpu().numpy()}")

        if step > longest_episode:
            longest_episode = step
            print(f"New longest episode: {longest_episode} steps")

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        losses.append(episode_loss / step if step > 0 else 0)
        avg_q_values.append(episode_q_value / step if step > 0 else 0)

        if episode_reward > best_reward:
            best_reward = episode_reward
            episodes_without_improvement = 0
        else:
            episodes_without_improvement += 1

        if episodes_without_improvement >= patience:
            print(f"Early stopping triggered after {episode + 1} episodes")
            break

        print(f"Episode {episode + 1} complete")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Snake Length: {snake_length}")
        print(f"Epsilon: {model.epsilon:.4f}")
        print(f"Steps: {step}")
        print(f"Average Loss: {losses[-1]:.4f}")
        print(f"Average Q-value: {avg_q_values[-1]:.4f}")
        print(f"Learning Rate: {model.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Priority Beta: {model.priority_beta:.4f}")

        # Update plots
        rewards_plot.set_data(range(len(episode_rewards)), episode_rewards)
        lengths_plot.set_data(range(len(episode_lengths)), episode_lengths)
        losses_plot.set_data(range(len(losses)), losses)
        q_values_plot.set_data(range(len(avg_q_values)), avg_q_values)

        window_size = min(100, len(episode_rewards))
        rewards_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        lengths_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        losses_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        q_values_avg = np.convolve(avg_q_values, np.ones(window_size)/window_size, mode='valid')

        rewards_avg_plot.set_data(range(window_size-1, len(episode_rewards)), rewards_avg)
        lengths_avg_plot.set_data(range(window_size-1, len(episode_lengths)), lengths_avg)
        losses_avg_plot.set_data(range(window_size-1, len(losses)), losses_avg)
        q_values_avg_plot.set_data(range(window_size-1, len(avg_q_values)), q_values_avg)

        for ax in (ax1, ax2, ax3, ax4):
            ax.relim()
            ax.autoscale_view()

        plt.draw()
        plt.pause(0.1)

        if (episode + 1) % checkpoint_interval == 0:
            model.save(f"snake/models/snake_{model_type}_model_episode_{episode + 1}.pth")

        model.update_epsilon()
        model.update_scheduler()

        # Add this line to pause after each episode
        #input("Press Enter to continue to the next episode...")
        print("\n")

    print("\nTraining complete.")
    model.save(f"snake/models/snake_{model_type}_model_final.pth")

    plt.ioff()  # Turn off interactive mode
    plt.show()

    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Rewards')
    window_size = 100
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg, label='Moving Average (100 episodes)')
    ax1.set_title('Episode Rewards During Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()

    # Plot episode lengths
    ax2.plot(episode_lengths, alpha=0.3, label='Episode Lengths')
    length_moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
    ax2.plot(range(window_size-1, len(episode_lengths)), length_moving_avg, label='Moving Average (100 episodes)')
    ax2.set_title('Episode Lengths During Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()

    # Plot losses
    ax3.plot(losses, alpha=0.3, label='Loss')
    loss_moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    ax3.plot(range(window_size-1, len(losses)), loss_moving_avg, label='Moving Average (100 episodes)')
    ax3.set_title('Loss During Training')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.legend()

    # Plot average Q-values
    ax4.plot(avg_q_values, alpha=0.3, label='Avg Q-value')
    q_moving_avg = np.convolve(avg_q_values, np.ones(window_size)/window_size, mode='valid')
    ax4.plot(range(window_size-1, len(avg_q_values)), q_moving_avg, label='Moving Average (100 episodes)')
    ax4.set_title('Average Q-values During Training')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Avg Q-value')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()

if __name__ == "__main__":
    train()  