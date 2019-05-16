import gym
import keras
import random
import numpy as np
from typing import List
from src.util import exponential_decay
import matplotlib.pyplot as plt

class dqn:
    def __init__(self, env: gym.Env, model: keras.models,
                 exploration_rate: float, min_exploration_rate: float,
                 exploration_rate_decay: float, discount_factor: float):
        self.model = model
        self.env = env
        self.memory: List = []
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.discount_factor = discount_factor

    def fit(self, num_episodes: int, num_iterations: int,
            batch_size: int, render: bool = False) -> List[float]:
        results: List[float] = []

        for episode in range(num_episodes):
            state = self.env.reset()

            for iteration in range(num_iterations):

                if render:
                    self.env.render()

                action = self.act(state)

                next_state, reward, done, _ = self.env.step(action)

                self.remember(state, action, reward, next_state, done)

                state = next_state

                if done:

                    print(f"episode: {episode}/{num_episodes},"
                          f" score: {iteration}")
                    results.append(iteration)

                    break
            self.replay(batch_size)
        return results

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if random.uniform(0, 1) <= self.exploration_rate:
            action = self.env.action_space.sample()
        else:
            state_2d = np.atleast_2d(state)
            act_values = self.model.predict(state_2d)
            action = np.argmax(act_values[0])
        return action

    def replay(self, batch_size: int) -> None:
        size_memory = len(self.memory)
        if size_memory >= batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = random.sample(self.memory, size_memory)
        for state, action, reward, next_state, done in minibatch:
            state_2d = np.atleast_2d(state)

            target = reward
            if not done:
                next_state_2d = np.atleast_2d(next_state)
                target = reward + self.discount_factor * \
                         np.amax(self.model.predict(next_state_2d)[0])
            predicted = self.model.predict(state_2d)
            predicted[0][action] = target
            self.model.fit(state_2d, predicted, epochs=1, verbose=0)

        self.exploration_rate = exponential_decay(self.exploration_rate,
                                                  self.min_exploration_rate,
                                                  self.exploration_rate_decay)
