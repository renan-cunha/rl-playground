import gym
import keras
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.dnn_factory import create_dnn


class PolicyGradientAgent:
    def __init__(self, env: gym.Env, model: keras.Model):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.model = model
        self.env = env

        self.states: List[np.ndarray] = []
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

    def _append_sample(self, state: np.ndarray, action: int,
                       reward: float) -> None:
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def _train_episode(self, discount_factor: float, batch_size: int) -> None:
        episode_length = len(self.states)
        discounted_rewards = PolicyGradientAgent._discount_rewards(
            self.rewards,
            discount_factor)

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]
        self.model.fit(update_inputs, advantages, epochs=1, verbose=0,
                       batch_size=batch_size)
        self.states, self.actions, self.rewards = [], [], []

    def train(self, num_episodes: int, discount_factor: float,
              batch_size: int, render: bool = False) -> None:

        scores: List[float] = []
        episodes = []

        for e in range(num_episodes):
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            while not done:
                if render:
                    self.env.render()

                action = self._get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                self._append_sample(state, action, reward)

                score += reward
                state = next_state

                if done:
                    self._train_episode(discount_factor, batch_size)
                    scores.append(score)
                    episodes.append(e)
                    plt.plot(episodes, scores, "b")
                    plt.savefig("cartpole_reinforce.png")
                    print(f"episode:{e} scores:{score}")


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    state_size = env.observation_space.shape[0]
    lr = 0.001
    loss = "categorical_crossentropy"
    num_hidden_layers = 2
    num_hidden_neurons = 24
    action_size = env.action_space.n
    activation_function = "softmax"

    dnn = create_dnn(input_length=state_size, learning_rate=lr, loss=loss,
                     num_hidden_layers=num_hidden_layers,
                     num_hidden_neurons=num_hidden_neurons,
                     output_length=action_size,
                     output_layer_actionvation_function=activation_function)

    policy_agent = PolicyGradientAgent(env, dnn)
    policy_agent.train(1000, 0.99, 32)