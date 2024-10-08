import torch
import numpy as np
import pygame
import matplotlib.pyplot as plt
from snake_env import SnakeEnv
from algos.ConvDQN import ConvDQN
import cv2
import argparse

def visualize_model(model_path, num_episodes=5, save_video=False, display_realtime=True):
    env = SnakeEnv(render_mode='infer')
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    dqn = ConvDQN(input_shape, n_actions)
    dqn.load(model_path)
    
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('snake_gameplay.mp4', fourcc, 10.0, (env.width, env.height))

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        food = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(dqn.device)
            action = dqn.select_action(state_tensor, epsilon=0)  # Use greedy policy
            state, reward, done, _ = env.step(action.item())
            
            total_reward += reward
            if reward > 2:
                food += 1

            steps += 1
            
            frame = state_to_rgb(state)
            
            if save_video:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            if display_realtime:
                plt.clf()
                plt.imshow(frame)
                plt.title(f"Episode {episode + 1}, Step {steps}, Total Reward: {total_reward}, Food Eaten: {food}")
                plt.axis('off')
                plt.pause(0.1)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {steps}")

    if save_video:
        video.release()

    plt.show()

def state_to_rgb(state):
    # Assuming state has shape (4, 18, 18)
    # Channel 0: Snake body
    # Channel 1: Snake head
    # Channel 2: Food
    # Channel 3: Danger
    rgb_image = np.zeros((18, 18, 3), dtype=np.uint8)
    rgb_image[state[0] == 1] = [0, 255, 0]  # Snake body: Green
    rgb_image[state[1] == 1] = [0, 200, 0]  # Snake head: Dark Green
    rgb_image[state[2] == 1] = [255, 0, 0]  # Food: Red
    rgb_image[state[3] == 1] = [128, 128, 128]  # Danger: Gray
    return rgb_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained Snake AI model")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--save_video", action="store_true", help="Save gameplay as video")
    parser.add_argument("--no_display", action="store_true", help="Don't display gameplay in real-time")

    args = parser.parse_args()

    visualize_model(args.model_path, num_episodes=args.episodes, save_video=args.save_video, display_realtime=not args.no_display)
