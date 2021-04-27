import sys
import numpy as np
import pygame


def close_all():
    pygame.display.quit()
    pygame.quit()
    sys.exit()


def check_events(env, agent):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            close_all()
        elif event.type == pygame.KEYDOWN:
            check_keydown_event(event, env, agent)


def check_keydown_event(event, env, agent):
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
