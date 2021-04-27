import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import pygame
from pygame import Rect
from csettings import *


class GridworldMultiAgent(Env):

    def __init__(self, nb_agents=2, nb_resources=2, gridsize=5, nb_steps=50, reward_extracting=10.0, reward_else=-1.0,
                 screen=None):

        if screen is not None:
            self.screen = screen
        self.cell_pixels = WINDOW_PIXELS / gridsize

        self.nb_agents = nb_agents
        self.nb_actions = 5
        self.nb_resources = nb_resources
        self.gridsize = gridsize
        self.nb_steps = nb_steps
        self.step_nb = 0
        self.reward_extracting = reward_extracting
        self.reward_else = reward_else

        self.action_space = Discrete(self.nb_actions**self.nb_agents)
        self.observation_space = MultiDiscrete([self.gridsize]*2*self.nb_resources + [self.gridsize]*2*self.nb_agents)

        np.random.seed(1)
        # [x,y] 
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        # [x,y] 
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))

        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5
            action = [0]*self.nb_agents
            num = i
            index = -1
            while num > 0:
                action[index] = num % self.nb_actions
                num = num // self.nb_actions
                index -= 1
            self.action_map[i] = action

    def step(self, action: int):
        self.step_nb += 1
        done = False
        if self.step_nb == self.nb_steps:
            done = True

        for i, action in enumerate(self.action_map[action]):
            if action == UP:
                self.state_agent[i, 1] = self.state_agent[i, 1] - 1 if self.state_agent[i, 1] > 0 else 0            
            elif action == RIGHT:
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[i, 0] < self.gridsize - 1 else self.gridsize - 1
            elif action == DOWN:
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[i, 1] < self.gridsize - 1 else self.gridsize - 1
            elif action == LEFT:
                self.state_agent[i, 0] = self.state_agent[i, 0] - 1 if self.state_agent[i, 0] > 0 else 0 
                
        reward = self.reward_else
        collected_resources = []
        for i, resource in enumerate(self.state_resources):
            for agent in self.state_agent:
                if np.all(resource == agent):
                    reward += self.reward_extracting
                    collected_resources.append(i)
                    break

        for i in collected_resources:
            self.state_resources[i, :] = np.random.randint(self.gridsize, size=2)

        info = {}

        # print("reward: ", reward)
        return self.observe(), reward, done, info

    def reset(self):
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.step_nb = 0
        return self.observe()
    
    def render(self, mode='human'):
        # Render background
        self.screen.fill(BG_COLOR)

        # Draw resources
        for resource in self.state_resources:
            rect = Rect(resource[0] * self.cell_pixels, resource[1] * self.cell_pixels, self.cell_pixels,
                        self.cell_pixels)
            pygame.draw.rect(self.screen, RESOURCE_COLOR, rect)

        # Draw grid
        for i in range(self.gridsize + 1):
            # (screen, color, (x0, y0), (x1, y1), width)
            # Vertical line
            pygame.draw.line(self.screen, GRID_COLOR, (i*self.cell_pixels, 0),
                             (i*self.cell_pixels, WINDOW_PIXELS-1), 1)
            # Horizontal line
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * self.cell_pixels),
                             (WINDOW_PIXELS - 1, i * self.cell_pixels), 1)

        # Draw agents
        for agent in self.state_agent:
            # Figure out center
            center = (agent[0]*self.cell_pixels + self.cell_pixels/2, agent[1]*self.cell_pixels + self.cell_pixels/2)
            pygame.draw.circle(self.screen, AGENT_COLOR, center, self.cell_pixels/3)

        # Discard old frames and show the last one
        pygame.display.flip()

    def observe(self):
        return np.concatenate((self.state_resources.flatten(), self.state_agent.flatten()))
