import gym
import keras
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler


class PolicyGradientAgent:
    def __init__(self, env: gym.Env, model: keras.Model):
        self.state_size = env.action_space.shape[0]
        self.action_size = env.action_space.n
        self.model = model

        self.states: List[List[float]] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []

    def _get_action(self, state: np.ndarray) -> int:
        """Choses an action based on the output probability of the model"""
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    @staticmethod
    def _discount_rewards(rewards: List[float],
                          discount_factor: float) -> np.ndarray:
        discounted_rewards = np.zeros(len(rewards))
        running_add = 0.0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def _append_sample(self, state: List[float], action: int,
                       reward: float) -> None:
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def _train_episode(self, discount_factor: float) -> None:
        episode_length = len(self.states)
        discounted_rewards = PolicyGradientAgent._discount_rewards(
            self.rewards,
            discount_factor)
        discounted_rewards = StandardScaler().fit_transform(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]





