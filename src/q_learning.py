import numpy as np
from typing import List
import random
import gym


def best_action(q_table: np.ndarray, state: int) -> int:
    """Returns the best option"""
    return int(np.argmax(q_table[state]))


def run_with_best_action(q_table: np.ndarray, string_env: str) -> float:
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
                               actions: List[int] = None) -> float:
    """Returns the max expected future reward given the state and actions"""
    if actions is None:
        actions = list(range(q_table.shape[1]))
    return np.max(q_table[state, actions])


def delta_q_value(q_table: np.ndarray, action: int, current_reward: float,
                  old_state: int, new_state: int,
                  discount_rate: float) -> float:
    """Returns the change in Q value deltaQ(state, action)"""
    result = current_reward
    result += discount_rate * max_expected_future_reward(q_table, new_state)
    return result - q_table[old_state, action]


def new_q_value(q_table: np.ndarray, old_state: int, action: int,
                learning_rate: float, current_reward: float,
                new_state: int, discount_rate :float) -> float:
    """Returns the new q value for that state and action"""
    result = q_table[old_state, action]
    result += learning_rate * delta_q_value(q_table, action, current_reward,
                                            old_state, new_state,
                                            discount_rate)
    return result


def update_q_value(q_table: np.ndarray, state: int, action: int,
                   learning_rate: float, current_reward: float,
                   new_state: int, discount_rate: float) -> None:
    """Returns nothing, updates the q_value"""
    q_table[state, action] = new_q_value(q_table, state, action, learning_rate,
                                         current_reward, new_state,
                                         discount_rate)


def choose_action(q_table: np.ndarray, state: int,
                  exploration_rate: float) -> int:
    """Returns an action considering the exploration/exploitation tradeoff"""
    random_value: float = random.uniform(0, 1)
    action: int
    if random_value > exploration_rate:
        action = best_action(q_table, state)
    else:
        num_actions: int = q_table.shape[1]
        action = random.randint(0, num_actions-1)
    return action


def reduce_exploration_rate(exploration_rate: float,
                            min_exploration_rate: float,
                            exploration_rate_decay: float) -> float:
    """Uses exponential decay to reduce the exploration rate"""
    multiplier = np.exp(-exploration_rate_decay)
    result = min_exploration_rate
    result += (exploration_rate - min_exploration_rate)*multiplier
    return result


def q_learning(num_iterations: int, learning_rate: float,
               discount_rate: float, exploration_rate: float,
               min_exploration_rate: float, exploration_rate_decay: float,
               string_env: str, save_path: str = None) -> np.ndarray:
    """Train and update the q_table"""
    env = gym.make(string_env)
    state = env.reset()
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))

    total_rewards = []
    for i in range(num_iterations):
        action = choose_action(q_table, state, exploration_rate)

        new_state, reward, done, info = env.step(action)
        update_q_value(q_table, state, action, learning_rate, reward, new_state,
                       discount_rate)

        if done == True:
            new_state = env.reset()
            current_reward = run_with_best_action(q_table, string_env)
            total_rewards.append(current_reward)
            exploration_rate = reduce_exploration_rate(exploration_rate,
                                                       min_exploration_rate,
                                                       exploration_rate_decay)

        state = new_state

    if save_path:
        np.save(save_path, q_table)
    print(np.mean(np.array(total_rewards[-100:])))
    return q_table
