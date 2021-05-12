from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from rl.agents import DQNAgent
from rl.memory import SequentialMemory  # For experience replay!


def build_model(states, actions, h_nodes, h_act):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    for n, a in zip(h_nodes, h_act):
        model.add(Dense(n, activation=a))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions, tmu, policy, ml):
    memory = SequentialMemory(limit=ml, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=100,
                   target_model_update=tmu)
    return dqn


def get_agent_path(name):
    return "agents/{}/{}.h5f".format(name, name)


def get_training_path(name):
    return "agents/{}/{}_training.json".format(name, name)


def get_test_path(name, nb_episodes):
    return 'agents/{}/{}_test_{}episodes.txt'.format(name, name, nb_episodes)
