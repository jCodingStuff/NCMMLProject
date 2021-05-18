import sys
import numpy as np
import pygame
from gym_environment_ncml import *

frame_count = 0


def close_all():
    pygame.display.quit()
    pygame.quit()
    sys.exit()


def check_events(screen, env, agent):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            close_all()
        elif event.type == pygame.KEYDOWN:
            check_keydown_event(event, screen, env, agent)


def check_keydown_event(event, screen, env, agent):
    global frame_count
    if event.key == pygame.K_q:
        close_all()
    elif event.key == pygame.K_0:
        env.step(0)
    elif event.key == pygame.K_1:
        env.step(1)
    elif event.key == pygame.K_5:
        env.step(5)
    elif event.key == pygame.K_6:
        env.step(6)
    elif event.key == pygame.K_9:
        env.step(9)
    elif event.key == pygame.K_SPACE:
        env.step(np.random.randint(env.action_space.n))
    elif event.key == pygame.K_s:
        env.step(agent.forward(env.observe()))
    elif event.key == pygame.K_r:
        env.reset()
    elif event.key == pygame.K_f:
        frame_count += 1
        if isinstance(env, GridworldMultiAgentv1):
            name = 'frames/env1_{}b{}_a{}_r{}_rext{}_relse{}_frame{}.jpg'
            pygame.image.save(
                screen,
                name.format(env.gridsize, env.gridsize, env.nb_agents, env.nb_resources, env.reward_extracting,
                            env.reward_else, frame_count)
            )
        elif isinstance(env, GridworldMultiAgentv15):
            name = 'frames/env15_{}b{}_a{}_r{}_rext{}_relse{}_frame{}.jpg'
            pygame.image.save(
                screen,
                name.format(env.gridsize, env.gridsize, env.nb_agents, env.nb_resources, env.reward_extracting,
                            env.reward_else, frame_count)
            )
        elif isinstance(env, GridworldMultiAgentv2):
            name = 'frames/env2_{}b{}_a{}_ap{}_r{}_c{}_radius{}_rext{}_relse{}_alpha{}_beta{}_frame{}.jpg'
            pygame.image.save(
                screen,
                name.format(env.gridsize, env.gridsize, env.nb_agents, env.agent_power, env.nb_resources,
                            env.nb_civilians, env.radius, env.reward_extracting, env.reward_else, env.alpha,
                            env.beta, frame_count)
            )
        elif isinstance(env, GridworldMultiAgentv25):
            name = 'frames/env2_{}b{}_a{}_ap{}_r{}_c{}_radius{}_rext{}_relse{}_alpha{}_beta{}_frame{}.jpg'
            pygame.image.save(
                screen,
                name.format(env.gridsize, env.gridsize, env.nb_agents, env.agent_power, env.nb_resources,
                            env.nb_civilians, env.radius, env.reward_extracting, env.reward_else, env.alpha,
                            env.beta, frame_count)
            )
        elif isinstance(env, GridworldMultiAgentv3):
            name = 'frames/env3_{}b{}_a{}_ap{}_r{}_c{}_radius{}_rext{}_relse{}_alpha{}_beta{}_frame{}.jpg'
            pygame.image.save(
                screen,
                name.format(env.gridsize, env.gridsize, env.nb_agents, env.agent_power, env.nb_resources,
                            env.nb_civilians, env.radius, env.reward_extracting, env.reward_else, env.alpha,
                            env.beta, frame_count)
            )
        elif isinstance(env, GridworldMultiAgentv35):
            name = 'frames/env35_{}b{}_a{}_ap{}_r{}_c{}_radius{}_rext{}_relse{}_alpha{}_beta{}_frame{}.jpg'
            pygame.image.save(
                screen,
                name.format(env.gridsize, env.gridsize, env.nb_agents, env.agent_power, env.nb_resources,
                            env.nb_civilians, env.radius, env.reward_extracting, env.reward_else, env.alpha,
                            env.beta, frame_count)
            )
