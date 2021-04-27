import numpy as np
import scipy 
import matplotlib
import gym
from gym.spaces import Discrete, MultiDiscrete
import tensorflow as tf
import pygame
import rl


class GridworldMultiAgent(gym.Env):

    def __init__(self, nb_agents = 2, nb_resources = 2, gridsize = 5, nb_steps = 50, 
                reward_extracting = 10, reward_else = -1):

        self.nb_agents = nb_agents
        self.nb_actions = 5
        self.nb_resources = nb_resources
        self.gridsize = gridsize
        self.nb_steps = nb_steps
        self.step = 0
        self.reward_extracting = reward_extracting
        self.reward_else = reward_else

        self.action_space = Discrete(self.nb_actions**self.nb_agents)
        self.observation_space = MultiDiscrete([self.gridsize]*2*self.nb_resources 
                                                + [self.gridsize]*2*self.nb_agents)

        np.random.seed(1)
        # [x,y] 
        self.state_agent = np.random.randint(self.grid_size, size=(self.n_agents, 2))
        # [x,y] 
        self.state_resources = np.random.randint(self.grid_size, size=(self.nb_resources, 2)) 

        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5
            action = [0]*n_agents
            num = i
            index = -1
            while num > 0:
                action[index] = num % self.nb_actions
                num = num // self.nb_actions
                index -= 1
            self.action_map[i] = action

    def step(self,action):
        self.step += 1
        done = False
        if self.step == nb_steps:
            done = True

        for i,action in enumerate(self.action_map[action]):
            if action == 1:
                #UP
                self.state_agent[i, 1] = self.state_agent[i, 1] - 1 if self.state_agent[i, 1] > 0 else 0            
            elif action == 2:
                #RIGHT
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[i, 0] < self.gridsize -1 else self.gridsize -1
            elif action == 3:
                #DOWN
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[i, 1] < self.gridsize -1 else self.gridsize -1 
            elif action == 4:
                #LEFT
                self.state_agent[i, 0] = self.state_agent[i, 0] - 1 if self.state_agent[i, 0] > 0 else 0 
                
        reward = self.reward_else
        collected_resources = []
        for i,resource in enumerate(self.state_resources):
            for agent in self.state_agent:
                if np.all(resource==agent):
                    reward += self.reward_extracting
                    collected_resources.append(i)
                    break

        for i in collected_resources:
            self.state_resources[i,:] = np.random.randint(self.grid_size, size = 2)

        info={}

        observation = np.concatenate((self.state_resources.flatten(), self.state_agent.flatten()))

        print("reward: ", reward)
        return observation, reward, done, info 

                

    def reset(self):
        self.state_agent = np.random.randint(self.grid_size, size=(self.n_agents, 2))
        self.state_resources = np.random.randint(self.grid_size, size=(self.nb_resources, 2))
        self.step = 0
    
    def render(self):
