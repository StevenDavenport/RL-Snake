import gymnasium as gym
import numpy as np
import pygame

class SnakeEnv(gym.Env):
    def __init__(self, width=360, height=360, render_mode=False):
        super(SnakeEnv, self).__init__()
        
        self.width = width
        self.height = height
        self.grid_size = 20
        self.render_mode = render_mode

        self.n_rows = height // self.grid_size
        self.n_cols = width // self.grid_size

        self.action_space = gym.spaces.Discrete(3)  # 0: Forward, 1: Left, 2: Right
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4, self.n_rows, self.n_cols), dtype=np.float32)
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        
        self.reset()

    def reset(self):
        # Start at the center of the screen
        center_x = (self.n_cols // 2) * self.grid_size
        center_y = (self.n_rows // 2) * self.grid_size
        
        # Initialize snake with three segments, moving right
        self.snake = [
            (center_x, center_y),
            (center_x - self.grid_size, center_y),
            (center_x - 2 * self.grid_size, center_y)
        ]
        self.direction = 1  # 0: up, 1: right, 2: down, 3: left
        self.score = 0
        self.food = self._generate_food()
        return self._get_obs()

    def step(self, action):
        print(f"Received action: {action}")
        # Convert relative action to absolute direction
        if action == 0:  # Forward
            new_direction = self.direction
        elif action == 1:  # Left
            new_direction = (self.direction - 1) % 4
        else:  # Right
            new_direction = (self.direction + 1) % 4

        print(f"New direction: {new_direction}")
        # Update direction
        self.direction = new_direction

        # Store the previous head position
        prev_head = self.snake[0]

        # Move snake
        head_x, head_y = self.snake[0]
        if self.direction == 0:
            head_y -= self.grid_size
        elif self.direction == 1:
            head_x += self.grid_size
        elif self.direction == 2:
            head_y += self.grid_size
        elif self.direction == 3:
            head_x -= self.grid_size
        
        # Check for wall collision
        if head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height:
            return self._get_obs(), -100, True, {'collision': 'wall', 'snake_length': len(self.snake)}

        self.snake.insert(0, (head_x, head_y))

        # Initialize reward
        reward = 0

        # Living penalty
        reward -= 0.01

        # Check if food is eaten
        if self.snake[0] == self.food:
            self.score += 1
            reward += 50  # High reward for eating food
            self.food = self._generate_food()
        else:
            self.snake.pop()

        # Check for self-collision
        if self.snake[0] in self.snake[1:]:
            return self._get_obs(), -100, True, {'collision': 'self', 'snake_length': len(self.snake)}

        # Reward for moving towards food, penalize for moving away
        prev_dist = self._manhattan_distance(prev_head, self.food)
        curr_dist = self._manhattan_distance(self.snake[0], self.food)
        if curr_dist < prev_dist:
            reward += 1
        else:
            reward -= 1  

        # Scaling survival reward
        snake_length = len(self.snake)
        survival_reward = 0.01 * (snake_length ** 1.5)  # Exponential scaling
        reward += survival_reward

        info = {
            'snake_length': len(self.snake)
        }

        return self._get_obs(), reward, False, info

    def render(self):
        if self.render_mode is None:
            return None

        surface = pygame.Surface((self.width, self.height))
        surface.fill((0, 0, 0))  # Fill with black

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(surface, (0, 255, 0), pygame.Rect(segment[0], segment[1], self.grid_size, self.grid_size))

        # Draw food
        pygame.draw.rect(surface, (255, 0, 0), pygame.Rect(self.food[0], self.food[1], self.grid_size, self.grid_size))

        if self.render_mode == 'human':
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(10)
            return self.screen
        elif self.render_mode in ['rgb_array', 'infer']:
            return pygame.surfarray.array3d(surface).transpose((1, 0, 2))

    def _get_obs(self):
        obs = np.zeros((4, self.n_rows, self.n_cols), dtype=np.float32)
        
        # Snake body
        for segment in self.snake:
            x, y = segment
            obs[0, y // self.grid_size, x // self.grid_size] = 1.0
        
        # Snake head
        head_x, head_y = self.snake[0]
        obs[1, head_y // self.grid_size, head_x // self.grid_size] = 1.0
        
        # Food
        food_x, food_y = self.food
        obs[2, food_y // self.grid_size, food_x // self.grid_size] = 1.0
        
        # Danger
        next_x, next_y = self._get_next_head_position()
        if (0 <= next_x < self.width and 0 <= next_y < self.height):
            obs[3, next_y // self.grid_size, next_x // self.grid_size] = 1.0
        
        return obs

    def _get_next_head_position(self):
        head_x, head_y = self.snake[0]
        if self.direction == 0:  # Up
            return head_x, head_y - self.grid_size
        elif self.direction == 1:  # Right
            return head_x + self.grid_size, head_y
        elif self.direction == 2:  # Down
            return head_x, head_y + self.grid_size
        else:  # Left
            return head_x - self.grid_size, head_y

    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.width // self.grid_size) * self.grid_size,
                    np.random.randint(0, self.height // self.grid_size) * self.grid_size)
            if food not in self.snake:
                return food

    def _is_collision(self, point):
        # Check if point is on the boundaries
        if point[0] < 0 or point[0] >= self.width or point[1] < 0 or point[1] >= self.height:
            return True
        # Check if point is on the snake body
        return point in self.snake[1:]

    def close(self):
        if self.screen is not None:
            pygame.quit()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
