import gym
import numpy as np
from typing import List


def max_expected_future_reward(q_table: np.ndarray, state: int,
                               actions: List[int] = None):
    """Returns the max expected future reward given the state and actions"""
    if actions is None:
        actions = list(range(q_table.shape[1]))
    return np.max(q_table[state, actions])


def delta_q_value(q_table, action, current_reward, old_state, new_state,
                  discount_rate):
    """Returns the change in Q value deltaQ(state, action)"""
    result = current_reward
    result += discount_rate * max_expected_future_reward(q_table, new_state)
    return result - q_table[old_state, action]


def new_q_value(q_table, old_state, action, learning_rate, current_reward,
                new_state, discount_rate):
    """Returns the new q value for that state and action"""
    result = q_table[old_state, action]
    result += learning_rate * delta_q_value(q_table, action, current_reward,
                                            old_state, new_state,
                                            discount_rate)
    return result


def update_q_value(q_table, state, action, learning_rate, current_reward,
                   new_state, discount_rate):
    """Returns nothing, updates the q_value"""
    q_table[state, action] = new_q_value(q_table, state, action, learning_rate,
                                         current_reward, new_state,
                                         discount_rate)


def q_learning(env, num_iterations, q_table, learning_rate, discount_rate):
    """Train and update the q_table"""
    state = env.reset()
    for i in range(num_iterations):
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        update_q_value(q_table, state, action, learning_rate, reward, new_state,
                       discount_rate)
        if done == True:
            new_state = env.reset()
        state = new_state


if __name__ == "__main__":
    env = gym.make("Taxi-v2")
    num_iterations = 1000000
    learning_rate = 1
    discount_rate = 1

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    q_learning(env, num_iterations, q_table, learning_rate, discount_rate)
    np.save("q_table.npy", q_table)


