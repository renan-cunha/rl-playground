import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class MarsRover():
    def __init__(self, initial_state: int):
        self.rewards = [1.0, 0, 0, 0, 0, 0, 10.0]
        self.state_space = list(range(7))
        self.transition_probability = [
                [0.6, 0.4, 0,   0,   0,   0,   0],
                [0.4, 0.2, 0.4, 0,   0,   0,   0],
                [0,   0.4, 0.2, 0.4, 0,   0,   0],
                [0,   0,   0.4, 0.2, 0.4, 0,   0],
                [0,   0,   0,   0.4, 0.2, 0.4, 0],
                [0,   0,   0,   0,   0.4, 0.2, 0.4],
                [0,   0,   0,   0,   0,   0.4, 0.6]
                ]
        self.state = initial_state

    def step(self, new_state: int) -> Tuple[int, int]:
        reward = self.rewards[new_state]
        self.state = new_state
        return reward, new_state

    def new_random_state(self) -> int:
        state = self.state
        state_space = self.state_space
        actions_probability = self.transition_probability[state]
        return np.random.choice(state_space, p=actions_probability)

    def reset(self, state: int) -> None:
        self.state = state


def monte_carlo(rover: MarsRover, state: int, time: int,
                episodes: int, 
                gamma: float) -> Tuple[float, float, float]:
    """Return value function of environment using monte carlo
    simulation"""
    rewards = []

    for i in range(episodes):
       
        rover.reset(state)
        current_reward = rover.rewards[state]
       
        for j in range(time):
            # episode
            new_state = rover.new_random_state()
            reward, _ = rover.step(new_state)
            current_reward = current_reward + reward*gamma**(j+1)

        rewards.append(current_reward)
   
    rewards = np.array(rewards)
    return rewards.mean(), max(rewards), min(rewards)


def dynamic_programming(rover: MarsRover, time:int,
                        gamma: float) -> List[float]:
    value = np.zeros(len(rover.state_space))
    print(value.shape)
    for i in range(time):
        for state in rover.state_space:
            transition_prob = rover.transition_probability[state]
            transition_prob = np.array(transition_prob)
            summation = gamma*sum(transition_prob*value)
            value[state] = rover.rewards[state] + summation
    return value


def iterative(rover: MarsRover, epsilon: float, 
              gamma: float) -> List[float]:
    inf = [float("inf")]*len(rover.state_space)
    value = np.array(inf)
    value_line = np.zeros(len(rover.state_space))
    inf = float("inf")
    while np.linalg.norm(abs(value-value_line), ord=inf) > epsilon:
        value = value_line.copy()
        for state in rover.state_space:
            transition_prob = rover.transition_probability[state]
            transition_prob = np.array(transition_prob)
            summation = gamma*sum(transition_prob*value)
            value_line[state] = rover.rewards[state] + summation
        print(value_line)
        print(value)
    return value_line

rover = MarsRover(0)

# Monte Carlo
results_monte = np.empty((3, 7))
for i in range(7):
    results_monte[:, i] = monte_carlo(rover, i, 100, 1000, 0.5)

# Dynamic
results_dynamic = dynamic_programming(rover, 1000, 0.5)

# iterative
results_iterative = iterative(rover, 0.01, 0.5)

plt.subplot(3,1,1)
plt.boxplot(results_monte)
plt.title("Monte Carlo")
plt.subplot(3,1,2)
plt.boxplot(results_dynamic.reshape(1,7))
plt.title("Dynamic Programming")
plt.subplot(3,1,3)
plt.boxplot(results_iterative.reshape(1,7))
plt.title("Iterative Algorithm")
plt.show()
