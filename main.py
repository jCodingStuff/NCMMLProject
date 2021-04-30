import pygame
from tensorflow.keras.optimizers import Adam
from rl.policy import BoltzmannQPolicy
from gym_environment_ncml import GridworldMultiAgent
from csettings import *
from learning import *
import ui_func as ui

pygame.init()
screen = pygame.display.set_mode((WINDOW_PIXELS, WINDOW_PIXELS))
pygame.display.set_caption('Resource Extraction Game')

env = GridworldMultiAgent(gridsize=5, nb_agents=2, nb_resources=2, screen=screen)

states = env.observation_space.shape[0]
actions = env.action_space.n
model = build_model(states, actions, [30, 30], ['relu', 'relu'])
print(model.summary())
dqn = build_agent(model, actions, 0.01, BoltzmannQPolicy(), 50000)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Load weights
dqn.load_weights('agents/dqn_3030_trial.h5f')

while True:
    ui.check_events(env, dqn)
    env.render()
