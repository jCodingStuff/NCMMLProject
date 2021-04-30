import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import pygame
from pygame import Rect
from csettings import *


# Resource extraction game environment class
class GridworldMultiAgent(Env):

    # Initialization function (constructor)
    def __init__(self, nb_agents=2, nb_resources=2, gridsize=5, nb_steps=50, reward_extracting=10.0, reward_else=-1.0,
                 screen=None):
        # nb_agents:            Number of agents
        # nb_resources:         Number of resources
        # gridsize:             Size of square grid
        # nb_steps:             Number of steps per episode
        # reward_extracting:    Resource extraction reward value
        # reward_else:          Default reward value
        # screen:               Pygame graphics rendering object

        # Activate graphics if specified
        if screen is not None:
            self.screen = screen
        # Compute cell pixel size
        self.cell_pixels = WINDOW_PIXELS / gridsize

        # Set number of possible actions and reset step number
        self.nb_actions = 5
        self.step_nb = 0
        
        # Set environment variables
        self.nb_agents = nb_agents
        self.nb_resources = nb_resources
        self.gridsize = gridsize
        self.nb_steps = nb_steps
        self.reward_extracting = reward_extracting
        self.reward_else = reward_else

        # Set up action and observation spaces
        self.action_space = Discrete(self.nb_actions**self.nb_agents)
        self.observation_space = MultiDiscrete([self.gridsize]*2*self.nb_resources + [self.gridsize]*2*self.nb_agents)

        # Set random seed for testing
        np.random.seed(1)
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))

        # Map action space to the base of possible actions, so that every digit corresponds to the action of one agent
        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5 (possible actions)
            action = [0]*self.nb_agents
            num = i
            index = -1
            while num > 0:
                action[index] = num % self.nb_actions
                num = num // self.nb_actions
                index -= 1
            self.action_map[i] = action

    # Step function
    def step(self, action: int):
        # Update position of each agent according to action map and grid boundaries
        for i, action in enumerate(self.action_map[action]):
            if action == UP:
                self.state_agent[i, 1] = self.state_agent[i, 1] - 1 if self.state_agent[i, 1] > 0 else 0            
            elif action == RIGHT:
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[i, 0] < self.gridsize - 1 else self.gridsize - 1
            elif action == DOWN:
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[i, 1] < self.gridsize - 1 else self.gridsize - 1
            elif action == LEFT:
                self.state_agent[i, 0] = self.state_agent[i, 0] - 1 if self.state_agent[i, 0] > 0 else 0 

        # Add default reward to total reward and create temporary list for extracted resources
        reward = self.reward_else
        extracted_resources = []
        # Check whether any agent is positioned upon a resource
        for i, resource in enumerate(self.state_resources):
            for agent in self.state_agent:
                # If the resource and agent have the same coordinates [x,y] ...
                if np.all(resource == agent):
                    # ... add resource extraction reward to total reward and mark resource as extracted
                    reward += self.reward_extracting
                    extracted_resources.append(i)
                    break

        # Randomize new coordinates [x,y] for extracted resources
        for i in extracted_resources:
            self.state_resources[i, :] = np.random.randint(self.gridsize, size=2)

        # Increase step by 1
        self.step_nb += 1
        done = False
        # If episode step limit is reached, finish episode
        if self.step_nb == self.nb_steps:
            done = True
        info = {}

        # print("reward: ", reward)
        return self.observe(), reward, done, info

    # Environment reset function
    def reset(self):
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        # Reset step number
        self.step_nb = 0
        return self.observe()

    # Graphics rendering function
    def render(self, mode='human'):
        # Render background
        self.screen.fill(BG_COLOR)

        # Draw resources as cell-sized green squares
        for resource in self.state_resources:
            rect = Rect(resource[0] * self.cell_pixels, resource[1] * self.cell_pixels, self.cell_pixels,
                        self.cell_pixels)
            pygame.draw.rect(self.screen, RESOURCE_COLOR, rect)

        # Draw cell grid as grey bars
        for i in range(self.gridsize + 1):
            # (screen, color, (x0, y0), (x1, y1), width)
            # Vertical line
            pygame.draw.line(self.screen, GRID_COLOR, (i*self.cell_pixels, 0),
                             (i*self.cell_pixels, WINDOW_PIXELS-1), 1)
            # Horizontal line
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * self.cell_pixels),
                             (WINDOW_PIXELS - 1, i * self.cell_pixels), 1)

        # Draw agents as blue circles
        for agent in self.state_agent:
            # Compute center
            center = (agent[0]*self.cell_pixels + self.cell_pixels/2, agent[1]*self.cell_pixels + self.cell_pixels/2)
            pygame.draw.circle(self.screen, AGENT_COLOR, center, self.cell_pixels/3)

        # Discard old frames and show the last one
        pygame.display.flip()

    # Observation generation function
    def observe(self):
        return np.concatenate((self.state_resources.flatten(), self.state_agent.flatten()))
