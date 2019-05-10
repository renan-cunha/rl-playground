import numpy as np
import matplotlib.pyplot as plt
from typing import List
import gym


def best_action(q_table: np.ndarray, state: int):
    """Returns the best option"""
    return np.argmax(q_table[state])


def run_with_best_action(q_table, string_env):
    """Makes a run with a q_table and outputs the accumulated reward"""
    env = gym.make(string_env)
    state = env.reset()
    total_reward = 0
    while True:
        action = best_action(q_table, state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


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


def q_learning(env, num_iterations, learning_rate, discount_rate, string_env,
               save_path = None):
    """Train and update the q_table"""
    state = env.reset()
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))

    total_rewards = []
    for i in range(num_iterations):
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        update_q_value(q_table, state, action, learning_rate, reward, new_state,
                       discount_rate)

        if done == True:
            new_state = env.reset()
            current_reward = run_with_best_action(q_table, string_env)
            total_rewards.append(current_reward)
            plt.plot(total_rewards)
            plt.pause(0.01)

        state = new_state

    if save_path:
        np.save(save_path, q_table)

    plt.show()