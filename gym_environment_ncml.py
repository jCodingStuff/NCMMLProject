import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, Box
import pygame
from pygame import Rect
from csettings import *


# Resource extraction game environment class, v.1
class GridworldMultiAgentv1(Env):

    # Initialization function (constructor)
    def __init__(self, nb_agents=2, nb_resources=2, gridsize=5, nb_steps=50, reward_extracting=10.0, reward_else=-1.0,
                 seed=1, screen=None, debug=False):
        # nb_agents:            Number of agents
        # nb_resources:         Number of resources
        # gridsize:             Size of square grid
        # nb_steps:             Number of steps per episode
        # reward_extracting:    Resource extraction reward value
        # reward_else:          Default reward value
        # screen:               Pygame graphics rendering object

        self.debug = debug

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
        self.action_space = Discrete(self.nb_actions ** self.nb_agents)
        self.observation_space = MultiDiscrete(
            [self.gridsize] * 2 * self.nb_resources + [self.gridsize] * 2 * self.nb_agents)

        # Set random seed for testing
        np.random.seed(seed)
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))

        # Map action space to the base of possible actions, so that every digit corresponds to the action of one agent
        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5 (possible actions)
            action = [0] * self.nb_agents
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
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[
                                                                           i, 0] < self.gridsize - 1 else self.gridsize - 1
            elif action == DOWN:
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[
                                                                           i, 1] < self.gridsize - 1 else self.gridsize - 1
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

        observation = self.observe()

        if self.debug:
            print("Reward:", reward)
            print('Observation:', observation)

        return observation, reward, done, info

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
            pygame.draw.line(self.screen, GRID_COLOR, (i * self.cell_pixels, 0),
                             (i * self.cell_pixels, WINDOW_PIXELS - 1), 1)
            # Horizontal line
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * self.cell_pixels),
                             (WINDOW_PIXELS - 1, i * self.cell_pixels), 1)

        # Draw agents as blue circles
        for agent in self.state_agent:
            # Compute center
            center = (agent[0] * self.cell_pixels + self.cell_pixels / 2,
                      agent[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGENT_COLOR, center, self.cell_pixels / 3)

        # Discard old frames and show the last one
        pygame.display.flip()

    # Observation generation function
    def observe(self):
        return np.concatenate((self.state_resources.flatten(), self.state_agent.flatten()))


# Resource extraction game environment class, v.1
class GridworldMultiAgentv15(Env):

    # Initialization function (constructor)
    def __init__(self, nb_agents=2, nb_resources=2, gridsize=5, nb_steps=50, reward_extracting=10.0, reward_else=-1.0,
                 seed=1, screen=None, debug=False):
        # nb_agents:            Number of agents
        # nb_resources:         Number of resources
        # gridsize:             Size of square grid
        # nb_steps:             Number of steps per episode
        # reward_extracting:    Resource extraction reward value
        # reward_else:          Default reward value
        # screen:               Pygame graphics rendering object

        self.debug = debug

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
        self.action_space = Discrete(self.nb_actions ** self.nb_agents)
        self.observation_space = Box(np.zeros(2 * self.nb_resources + 2 * self.nb_agents),
                                     np.ones(2 * self.nb_resources + 2 * self.nb_agents))

        # Set random seed for testing
        np.random.seed(seed)
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))

        # Map action space to the base of possible actions, so that every digit corresponds to the action of one agent
        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5 (possible actions)
            action = [0] * self.nb_agents
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
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[
                                                                           i, 0] < self.gridsize - 1 else self.gridsize - 1
            elif action == DOWN:
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[
                                                                           i, 1] < self.gridsize - 1 else self.gridsize - 1
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

        observation = self.observe()

        if self.debug:
            print("Reward:", reward)
            print('Observation:', observation)

        return observation, reward, done, info

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
            pygame.draw.line(self.screen, GRID_COLOR, (i * self.cell_pixels, 0),
                             (i * self.cell_pixels, WINDOW_PIXELS - 1), 1)
            # Horizontal line
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * self.cell_pixels),
                             (WINDOW_PIXELS - 1, i * self.cell_pixels), 1)

        # Draw agents as blue circles
        for agent in self.state_agent:
            # Compute center
            center = (agent[0] * self.cell_pixels + self.cell_pixels / 2,
                      agent[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGENT_COLOR, center, self.cell_pixels / 3)

        # Discard old frames and show the last one
        pygame.display.flip()

    # Observation generation function
    def observe(self):
        return np.concatenate((self.state_resources.flatten(),
                               self.state_agent.flatten())).astype(float)/(self.gridsize-1)


###############################################################################


# Resource extraction game environment class, v.2
class GridworldMultiAgentv2(Env):

    # Initialization function (constructor)
    def __init__(self, nb_agents=2, agent_power=1, nb_resources=2, nb_civilians=5, gridsize=5, radius=1, nb_steps=50,
                 reward_extracting=10.0, alpha=6, beta=0, reward_else=-1.0, seed=1, screen=None, debug=False):
        # nb_agents:            Number of agents
        # nb_resources:         Number of resources
        # gridsize:             Size of square grid
        # nb_steps:             Number of steps per episode
        # reward_extracting:    Resource extraction reward value
        # reward_else:          Default reward value
        # screen:               Pygame graphics rendering object

        self.debug = debug

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
        self.agent_power = agent_power
        self.nb_resources = nb_resources
        self.nb_civilians = nb_civilians
        self.gridsize = gridsize
        self.radius = radius
        self.nb_steps = nb_steps
        self.reward_extracting = reward_extracting
        self.alpha = alpha
        self.beta = beta
        self.reward_else = reward_else

        # Set up action and observation spaces
        self.action_space = Discrete(self.nb_actions ** self.nb_agents)
        self.observation_space = MultiDiscrete(
            [self.gridsize] * 2 * self.nb_resources + [self.gridsize] * 2 * self.nb_agents +
            [self.gridsize] * 2 * self.nb_civilians)

        # Set random seed for testing
        np.random.seed(seed)
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))

        # Map action space to the base of possible actions, so that every digit corresponds to the action of one agent
        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5 (possible actions)
            action = [0] * self.nb_agents
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
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[
                                                                           i, 0] < self.gridsize - 1 else self.gridsize - 1
            elif action == DOWN:
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[
                                                                           i, 1] < self.gridsize - 1 else self.gridsize - 1
            elif action == LEFT:
                self.state_agent[i, 0] = self.state_agent[i, 0] - 1 if self.state_agent[i, 0] > 0 else 0

                # Add default reward to total reward and create temporary list for extracted resources
        reward = self.reward_else
        extracted_resources = []
        # Check whether any agent is positioned upon a resource
        for i, resource in enumerate(self.state_resources):
            nb_agents_radius = 0
            nb_civilians_radius = 0
            extracted = False
            for agent in self.state_agent:
                if (resource[0] - self.radius <= agent[0] <= resource[0] + self.radius and
                        resource[1] - self.radius <= agent[1] <= resource[1] + self.radius):
                    nb_agents_radius += 1
                    # If the resource and agent have the same coordinates [x,y] ...
                    if np.all(resource == agent):
                        # ... add resource extraction reward to total reward and mark resource as extracted
                        extracted = True
                        extracted_resources.append(i)
            if extracted:
                for civilian in self.state_civilians:
                    if (resource[0] - self.radius <= civilian[0] <= resource[0] + self.radius and
                            resource[1] - self.radius <= civilian[1] <= resource[1] + self.radius):
                        nb_civilians_radius += 1
                reward += self.reward_extracting
                riot_size = (nb_civilians_radius - self.agent_power * nb_agents_radius)
                if riot_size > 0:
                    reward -= self.alpha * riot_size
                else:
                    reward -= self.beta * riot_size

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

        observation = self.observe()

        if self.debug:
            print("Reward:", reward)
            print('Observation:', observation)

        return observation, reward, done, info

    # Environment reset function
    def reset(self):
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))
        # Reset step number
        self.step_nb = 0
        return self.observe()

    # Graphics rendering function
    def render(self, mode='human'):
        # Render background
        self.screen.fill(BG_COLOR)

        # Draw resources and their radii as cell-sized green squares
        for resource in self.state_resources:
            for x in range(resource[0] - self.radius, resource[0] + self.radius + 1):
                for y in range(resource[1] - self.radius, resource[1] + self.radius + 1):
                    rect = Rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels,
                                self.cell_pixels)
                    pygame.draw.rect(self.screen, RADIUS_COLOR, rect)

        for resource in self.state_resources:
            rect = Rect(resource[0] * self.cell_pixels, resource[1] * self.cell_pixels, self.cell_pixels,
                        self.cell_pixels)
            pygame.draw.rect(self.screen, RESOURCE_COLOR, rect)

        # Draw cell grid as grey bars
        for i in range(self.gridsize + 1):
            # (screen, color, (x0, y0), (x1, y1), width)
            # Vertical line
            pygame.draw.line(self.screen, GRID_COLOR, (i * self.cell_pixels, 0),
                             (i * self.cell_pixels, WINDOW_PIXELS - 1), 1)
            # Horizontal line
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * self.cell_pixels),
                             (WINDOW_PIXELS - 1, i * self.cell_pixels), 1)

        # Draw agents as blue circles
        for agent in self.state_agent:
            # Compute center
            center = (agent[0] * self.cell_pixels + self.cell_pixels / 2,
                      agent[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGENT_COLOR, center, self.cell_pixels / 3)

        # Draw agents as blue circles
        for civilian in self.state_civilians:
            # Compute center
            center = (civilian[0] * self.cell_pixels + self.cell_pixels / 2,
                      civilian[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGITATED_COLOR, center, self.cell_pixels / 5)

        # Discard old frames and show the last one
        pygame.display.flip()

    # Observation generation function
    def observe(self):
        return np.concatenate((self.state_resources.flatten(), self.state_agent.flatten(),
                               self.state_civilians.flatten()))


# Resource extraction game environment class, v.2.5
class GridworldMultiAgentv25(Env):

    # Initialization function (constructor)
    def __init__(self, nb_agents=2, agent_power=1, nb_resources=2, nb_civilians=5, gridsize=5, radius=1, nb_steps=50,
                 reward_extracting=10.0, alpha=6, beta=0, reward_else=-1.0, seed=1, screen=None, debug=False):
        # nb_agents:            Number of agents
        # nb_resources:         Number of resources
        # gridsize:             Size of square grid
        # nb_steps:             Number of steps per episode
        # reward_extracting:    Resource extraction reward value
        # reward_else:          Default reward value
        # screen:               Pygame graphics rendering object

        # Debug mode
        self.debug = debug

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
        self.agent_power = agent_power
        self.nb_resources = nb_resources
        self.nb_civilians = nb_civilians
        self.gridsize = gridsize
        self.radius = radius
        self.nb_steps = nb_steps
        self.reward_extracting = reward_extracting
        self.alpha = alpha
        self.beta = beta
        self.reward_else = reward_else

        # Set up action and observation spaces
        self.action_space = Discrete(self.nb_actions ** self.nb_agents)
        self.observation_space = Box(np.zeros(3 * self.nb_resources + 2 * self.nb_agents),
                                     np.ones(3 * self.nb_resources + 2 * self.nb_agents))

        # Set random seed for testing
        np.random.seed(seed)
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))

        # Map action space to the base of possible actions, so that every digit corresponds to the action of one agent
        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5 (possible actions)
            action = [0] * self.nb_agents
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
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[
                                                                           i, 0] < self.gridsize - 1 else self.gridsize - 1
            elif action == DOWN:
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[
                                                                           i, 1] < self.gridsize - 1 else self.gridsize - 1
            elif action == LEFT:
                self.state_agent[i, 0] = self.state_agent[i, 0] - 1 if self.state_agent[i, 0] > 0 else 0

                # Add default reward to total reward and create temporary list for extracted resources
        reward = self.reward_else
        extracted_resources = []
        # Check whether any agent is positioned upon a resource
        for i, resource in enumerate(self.state_resources):
            nb_agents_radius = 0
            nb_civilians_radius = 0
            extracted = False
            for agent in self.state_agent:
                if (resource[0] - self.radius <= agent[0] <= resource[0] + self.radius and
                        resource[1] - self.radius <= agent[1] <= resource[1] + self.radius):
                    nb_agents_radius += 1
                    # If the resource and agent have the same coordinates [x,y] ...
                    if np.all(resource == agent):
                        # ... add resource extraction reward to total reward and mark resource as extracted
                        extracted = True
                        extracted_resources.append(i)
            if extracted:
                for civilian in self.state_civilians:
                    if (resource[0] - self.radius <= civilian[0] <= resource[0] + self.radius and
                            resource[1] - self.radius <= civilian[1] <= resource[1] + self.radius):
                        nb_civilians_radius += 1
                reward += self.reward_extracting
                riot_size = (nb_civilians_radius - self.agent_power * nb_agents_radius)
                if riot_size > 0:
                    reward -= self.alpha * riot_size
                else:
                    reward -= self.beta * riot_size

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

        observation = self.observe()

        if self.debug:
            print("Reward:", reward)
            print('Observation:', observation)

        return observation, reward, done, info

    # Environment reset function
    def reset(self):
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))
        # Reset step number
        self.step_nb = 0
        return self.observe()

    # Graphics rendering function
    def render(self, mode='human'):
        # Render background
        self.screen.fill(BG_COLOR)

        # Draw resources and their radii as cell-sized green squares
        for resource in self.state_resources:
            for x in range(resource[0] - self.radius, resource[0] + self.radius + 1):
                for y in range(resource[1] - self.radius, resource[1] + self.radius + 1):
                    rect = Rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels,
                                self.cell_pixels)
                    pygame.draw.rect(self.screen, RADIUS_COLOR, rect)

        for resource in self.state_resources:
            rect = Rect(resource[0] * self.cell_pixels, resource[1] * self.cell_pixels, self.cell_pixels,
                        self.cell_pixels)
            pygame.draw.rect(self.screen, RESOURCE_COLOR, rect)

        # Draw cell grid as grey bars
        for i in range(self.gridsize + 1):
            # (screen, color, (x0, y0), (x1, y1), width)
            # Vertical line
            pygame.draw.line(self.screen, GRID_COLOR, (i * self.cell_pixels, 0),
                             (i * self.cell_pixels, WINDOW_PIXELS - 1), 1)
            # Horizontal line
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * self.cell_pixels),
                             (WINDOW_PIXELS - 1, i * self.cell_pixels), 1)

        # Draw agents as blue circles
        for agent in self.state_agent:
            # Compute center
            center = (
                agent[0] * self.cell_pixels + self.cell_pixels / 2, agent[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGENT_COLOR, center, self.cell_pixels / 3)

        # Draw agents as blue circles
        for civilian in self.state_civilians:
            # Compute center
            center = (civilian[0] * self.cell_pixels + self.cell_pixels / 2,
                      civilian[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGITATED_COLOR, center, self.cell_pixels / 5)

        # Discard old frames and show the last one
        pygame.display.flip()

    # Observation generation function
    def observe(self):
        norm_agents = self.state_agent.flatten().astype(float) / (self.gridsize - 1)
        norm_resources = []
        for resource in self.state_resources:
            nb_civilians_close = 0
            for civilian in self.state_civilians:
                if (resource[0] - self.radius <= civilian[0] <= resource[0] + self.radius and
                        resource[1] - self.radius <= civilian[1] <= resource[1] + self.radius):
                    nb_civilians_close += 1
            norm_resources += [resource[0] / (self.gridsize - 1), resource[1] / (self.gridsize - 1),
                               nb_civilians_close / self.nb_civilians]

        return np.concatenate((np.array(norm_resources), norm_agents))


###############################################################################

# Resource extraction game environment class, v.3.0
class GridworldMultiAgentv3(Env):

    # Initialization function (constructor)
    def __init__(self, nb_agents=2, agent_power=1, nb_resources=2, nb_civilians=5, gridsize=5, radius=1, nb_steps=50,
                 reward_extracting=10.0, alpha=6, beta=0, reward_else=-1.0, seed=1, screen=None, debug=False):
        # nb_agents:            Number of agents
        # nb_resources:         Number of resources
        # gridsize:             Size of square grid
        # nb_steps:             Number of steps per episode
        # reward_extracting:    Resource extraction reward value
        # reward_else:          Default reward value
        # screen:               Pygame graphics rendering object

        # Debug mode
        self.debug = debug

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
        self.agent_power = agent_power
        self.nb_resources = nb_resources
        self.nb_civilians = nb_civilians
        self.gridsize = gridsize
        self.radius = radius
        self.nb_steps = nb_steps
        self.reward_extracting = reward_extracting
        self.alpha = alpha
        self.beta = beta
        self.reward_else = reward_else

        # Set up action and observation spaces
        self.action_space = Discrete(self.nb_actions ** self.nb_agents)
        self.observation_space = Box(np.zeros(3 * self.nb_resources + 2 * self.nb_agents),
                                     np.ones(3 * self.nb_resources + 2 * self.nb_agents))

        # Set random seed for testing
        np.random.seed(seed)
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))

        # Map action space to the base of possible actions, so that every digit corresponds to the action of one agent
        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5 (possible actions)
            action = [0] * self.nb_agents
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
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[
                                                                           i, 0] < self.gridsize - 1 else self.gridsize - 1
            elif action == DOWN:
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[
                                                                           i, 1] < self.gridsize - 1 else self.gridsize - 1
            elif action == LEFT:
                self.state_agent[i, 0] = self.state_agent[i, 0] - 1 if self.state_agent[i, 0] > 0 else 0

        # Add default reward to total reward and create temporary list for extracted resources
        reward = self.reward_else
        # Check whether any agent is positioned upon a resource
        for i, resource in enumerate(self.state_resources):
            if not np.all(resource == -1):
                nb_agents_radius = 0
                nb_civilians_radius = 0
                extracted = False
                for agent in self.state_agent:
                    if (resource[0] - self.radius <= agent[0] <= resource[0] + self.radius and
                            resource[1] - self.radius <= agent[1] <= resource[1] + self.radius):
                        nb_agents_radius += 1
                        # If the resource and agent have the same coordinates [x,y] ...
                        if np.all(resource == agent):
                            # ... add resource extraction reward to total reward and mark resource as extracted
                            extracted = True
                if extracted:
                    for civilian in self.state_civilians:
                        if (resource[0] - self.radius <= civilian[0] <= resource[0] + self.radius and
                                resource[1] - self.radius <= civilian[1] <= resource[1] + self.radius):
                            nb_civilians_radius += 1
                    self.state_resources[i] = np.array([-1, -1])
                    reward += self.reward_extracting
                    riot_size = (nb_civilians_radius - self.agent_power * nb_agents_radius)
                    if riot_size > 0:
                        reward -= self.alpha * riot_size
                    else:
                        reward -= self.beta * riot_size

        # Increase step by 1
        self.step_nb += 1
        done = False
        # If episode step limit is reached or all resources extracted, finish episode
        if self.step_nb == self.nb_steps or np.all(self.state_resources.flatten() == -1):
            done = True
        info = {}

        observation = self.observe()

        if self.debug:
            print("Reward:", reward)
            print('Observation:', observation)

        return observation, reward, done, info

    # Environment reset function
    def reset(self):
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))
        # Reset step number
        self.step_nb = 0
        return self.observe()

    # Graphics rendering function
    def render(self, mode='human'):
        # Render background
        self.screen.fill(BG_COLOR)

        # Draw resources and their radii as cell-sized green squares
        for resource in self.state_resources:
            if not np.all(resource == -1):
                for x in range(resource[0] - self.radius, resource[0] + self.radius + 1):
                    for y in range(resource[1] - self.radius, resource[1] + self.radius + 1):
                        rect = Rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels,
                                    self.cell_pixels)
                        pygame.draw.rect(self.screen, RADIUS_COLOR, rect)

        for resource in self.state_resources:
            if not np.all(resource == -1):
                rect = Rect(resource[0] * self.cell_pixels, resource[1] * self.cell_pixels, self.cell_pixels,
                            self.cell_pixels)
                pygame.draw.rect(self.screen, RESOURCE_COLOR, rect)

        # Draw cell grid as grey bars
        for i in range(self.gridsize + 1):
            # (screen, color, (x0, y0), (x1, y1), width)
            # Vertical line
            pygame.draw.line(self.screen, GRID_COLOR, (i * self.cell_pixels, 0),
                             (i * self.cell_pixels, WINDOW_PIXELS - 1), 1)
            # Horizontal line
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * self.cell_pixels),
                             (WINDOW_PIXELS - 1, i * self.cell_pixels), 1)

        # Draw agents as blue circles
        for agent in self.state_agent:
            # Compute center
            center = (
                agent[0] * self.cell_pixels + self.cell_pixels / 2, agent[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGENT_COLOR, center, self.cell_pixels / 3)

        # Draw agents as blue circles
        for civilian in self.state_civilians:
            # Compute center
            center = (civilian[0] * self.cell_pixels + self.cell_pixels / 2,
                      civilian[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGITATED_COLOR, center, self.cell_pixels / 5)

        # Discard old frames and show the last one
        pygame.display.flip()

    # Observation generation function
    def observe(self):
        norm_agents = self.state_agent.flatten().astype(float) / (self.gridsize - 1)
        norm_resources = []
        for resource in self.state_resources:
            nb_civilians_close = 0
            if not np.all(resource == -1):
                for civilian in self.state_civilians:
                    if (resource[0] - self.radius <= civilian[0] <= resource[0] + self.radius and
                            resource[1] - self.radius <= civilian[1] <= resource[1] + self.radius):
                        nb_civilians_close += 1
            norm_resources += [resource[0] / (self.gridsize - 1), resource[1] / (self.gridsize - 1),
                               nb_civilians_close / self.nb_civilians]

        return np.concatenate((np.array(norm_resources), norm_agents))


# Resource extraction game environment class, v.3.5
class GridworldMultiAgentv35(Env):

    # Initialization function (constructor)
    def __init__(self, nb_agents=2, agent_power=1, nb_resources=2, nb_civilians=5, gridsize=5, radius=1, nb_steps=50,
                 reward_extracting=10.0, alpha=6, beta=0, reward_else=-1.0, seed=1, screen=None, debug=False):
        # nb_agents:            Number of agents
        # nb_resources:         Number of resources
        # gridsize:             Size of square grid
        # nb_steps:             Number of steps per episode
        # reward_extracting:    Resource extraction reward value
        # reward_else:          Default reward value
        # screen:               Pygame graphics rendering object

        # Debug mode
        self.debug = debug

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
        self.agent_power = agent_power
        self.nb_resources = nb_resources
        self.nb_civilians = nb_civilians
        self.gridsize = gridsize
        self.radius = radius
        self.nb_steps = nb_steps
        self.reward_extracting = reward_extracting
        self.alpha = alpha
        self.beta = beta
        self.reward_else = reward_else
        self.extracted_resources = np.array([False]*self.nb_resources)

        # Set up action and observation spaces
        self.action_space = Discrete(self.nb_actions ** self.nb_agents)
        self.observation_space = Box(np.zeros(4 * self.nb_resources + 2 * self.nb_agents),
                                     np.ones(4 * self.nb_resources + 2 * self.nb_agents))

        # Set random seed for testing
        np.random.seed(seed)
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))

        # Map action space to the base of possible actions, so that every digit corresponds to the action of one agent
        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5 (possible actions)
            action = [0] * self.nb_agents
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
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[
                                                                           i, 0] < self.gridsize - 1 else self.gridsize - 1
            elif action == DOWN:
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[
                                                                           i, 1] < self.gridsize - 1 else self.gridsize - 1
            elif action == LEFT:
                self.state_agent[i, 0] = self.state_agent[i, 0] - 1 if self.state_agent[i, 0] > 0 else 0

        # Add default reward to total reward and create temporary list for extracted resources
        reward = self.reward_else
        # Check whether any agent is positioned upon a resource
        for i, resource in enumerate(self.state_resources):
            if not self.extracted_resources[i]:
                nb_agents_radius = 0
                nb_civilians_radius = 0
                for agent in self.state_agent:
                    if (resource[0] - self.radius <= agent[0] <= resource[0] + self.radius and
                            resource[1] - self.radius <= agent[1] <= resource[1] + self.radius):
                        nb_agents_radius += 1
                        # If the resource and agent have the same coordinates [x,y] ...
                        if np.all(resource == agent):
                            # ... add resource extraction reward to total reward and mark resource as extracted
                            self.extracted_resources[i] = True
                if self.extracted_resources[i]:
                    for civilian in self.state_civilians:
                        if (resource[0] - self.radius <= civilian[0] <= resource[0] + self.radius and
                                resource[1] - self.radius <= civilian[1] <= resource[1] + self.radius):
                            nb_civilians_radius += 1
                    reward += self.reward_extracting
                    riot_size = (nb_civilians_radius - self.agent_power * nb_agents_radius)
                    if riot_size > 0:
                        reward -= self.alpha * riot_size
                    else:
                        reward -= self.beta * riot_size

        # Increase step by 1
        self.step_nb += 1
        done = False
        # If episode step limit is reached or all resources extracted, finish episode
        if self.step_nb == self.nb_steps or np.all(self.extracted_resources):
            done = True
        info = {}

        observation = self.observe()

        if self.debug:
            print("Reward:", reward)
            print('Observation:', observation)

        return observation, reward, done, info

    # Environment reset function
    def reset(self):
        self.extracted_resources = np.array([False] * self.nb_resources)
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))
        # Reset step number
        self.step_nb = 0
        return self.observe()

    # Graphics rendering function
    def render(self, mode='human'):
        # Render background
        self.screen.fill(BG_COLOR)

        # Draw resources and their radii as cell-sized green squares
        for i, resource in enumerate(self.state_resources):
            if not self.extracted_resources[i]:
                for x in range(resource[0] - self.radius, resource[0] + self.radius + 1):
                    for y in range(resource[1] - self.radius, resource[1] + self.radius + 1):
                        rect = Rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels,
                                    self.cell_pixels)
                        pygame.draw.rect(self.screen, RADIUS_COLOR, rect)

        for i, resource in enumerate(self.state_resources):
            if not self.extracted_resources[i]:
                rect = Rect(resource[0] * self.cell_pixels, resource[1] * self.cell_pixels, self.cell_pixels,
                            self.cell_pixels)
                pygame.draw.rect(self.screen, RESOURCE_COLOR, rect)

        # Draw cell grid as grey bars
        for i in range(self.gridsize + 1):
            # (screen, color, (x0, y0), (x1, y1), width)
            # Vertical line
            pygame.draw.line(self.screen, GRID_COLOR, (i * self.cell_pixels, 0),
                             (i * self.cell_pixels, WINDOW_PIXELS - 1), 1)
            # Horizontal line
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * self.cell_pixels),
                             (WINDOW_PIXELS - 1, i * self.cell_pixels), 1)

        # Draw agents as blue circles
        for agent in self.state_agent:
            # Compute center
            center = (
                agent[0] * self.cell_pixels + self.cell_pixels / 2, agent[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGENT_COLOR, center, self.cell_pixels / 3)

        # Draw agents as blue circles
        for civilian in self.state_civilians:
            # Compute center
            center = (civilian[0] * self.cell_pixels + self.cell_pixels / 2,
                      civilian[1] * self.cell_pixels + self.cell_pixels / 2)
            pygame.draw.circle(self.screen, AGITATED_COLOR, center, self.cell_pixels / 5)

        # Discard old frames and show the last one
        pygame.display.flip()

    # Observation generation function
    def observe(self):
        norm_agents = self.state_agent.flatten().astype(float) / (self.gridsize - 1)
        norm_resources = []
        for i, resource in enumerate(self.state_resources):
            nb_civilians_close = 0
            if not self.extracted_resources[i]:
                for civilian in self.state_civilians:
                    if (resource[0] - self.radius <= civilian[0] <= resource[0] + self.radius and
                            resource[1] - self.radius <= civilian[1] <= resource[1] + self.radius):
                        nb_civilians_close += 1
            norm_resources += [resource[0] / (self.gridsize - 1), resource[1] / (self.gridsize - 1),
                               int(self.extracted_resources[i]), nb_civilians_close / self.nb_civilians]

        return np.concatenate((np.array(norm_resources), norm_agents))
