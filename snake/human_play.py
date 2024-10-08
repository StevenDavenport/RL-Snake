import pygame
from snake_env import SnakeEnv

def human_play(num_episodes=5):
    env = SnakeEnv(render_mode="human")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"Episode {episode + 1} started. Use arrow keys to control the snake.")
        print("Left Arrow: Turn Left, Right Arrow: Turn Right, Up Arrow: Go Forward")

        env.render()  # Render initial state

        while not done:
            action = None  # No action by default
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 1  # Turn Left
                    elif event.key == pygame.K_RIGHT:
                        action = 2  # Turn Right
                    elif event.key == pygame.K_UP:
                        action = 0  # Go Forward

            if action is not None:
                state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

                env.render()

                if 'apple_eaten' in info and info['apple_eaten']:
                    print(f"Step {steps}: Apple eaten!")

            pygame.time.wait(200)  # Add a small delay for smoother gameplay

        print(f"Episode {episode + 1} finished. Total Reward: {total_reward}, Steps: {steps}")

    pygame.quit()

if __name__ == "__main__":
    human_play()