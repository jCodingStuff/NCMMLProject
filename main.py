import pygame
from tensorflow.keras.optimizers import Adam
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from gym_environment_ncml import *
from csettings import *
from learning import *
import ui_func as ui

pygame.init()
screen = pygame.display.set_mode((WINDOW_PIXELS, WINDOW_PIXELS))
pygame.display.set_caption('Resource Extraction Game')

env = GridworldMultiAgentv25(gridsize=5, nb_agents=2, nb_resources=2, alpha=10, beta=0, screen=screen, debug=True)

states = env.observation_space.shape[0]
actions = env.action_space.n
model = build_model(states, actions, [32, 16], ['relu', 'relu'])
# print(model.summary())
dqn = build_agent(model, actions, 0.01, EpsGreedyQPolicy(), 50000)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Load weights
dqn.load_weights(get_agent_path('dqn25_5b5_3236_adam_lr0.001_tmu0.01_ml50K_ns5M_eps0.1_a10'))

while True:
    ui.check_events(screen, env, dqn)
    env.render()
