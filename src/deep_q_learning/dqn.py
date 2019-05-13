import gym
import keras
import random
import numpy as np


class dqn:
    def __init__(self, env: gym.Env, model: keras.models):
        self.model = model
        self.env = env
        self.memory = []
        self.exploration_rate = 1.0

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def fit(self, num_episodes: int, exploration_rate: float,
            exploration_rate_decay: float,
            min_exploration_rate: float) -> None:
        self.exploration_rate = exploration_rate

    def act(self, state) -> int:
        if random.uniform(0, 1) <= self.exploration_rate:
            action = self.env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            action = np.argmax(act_values[0])
        return action



    def replay(self, batch_size) -> None:
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.exploration_rate * \
                         np.amax(self.model.predict(next_state)[0])
            predicted = self.model.predict(state)
            predicted[0][action] = target
            self.model.fit(state, predicted, epochs=1, verbose=0)





