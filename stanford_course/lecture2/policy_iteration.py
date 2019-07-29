import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class MarsRover():
    def __init__(self, initial_state: int):
        self.rewards = [1.0, 0, 0, 0, 0, 0, 10.0]
        self.state_space = list(range(7))
        self.action_space = list(range(2))
        self.state = initial_state

    def step(self, action: int) -> Tuple[int, int]:
        if action == 1:
            new_state = min([self.state+1, self.state_space[-1]]) # state has limit
        elif action == 0:
            new_state = max([0, self.state-1]) # state space has lower bound
        else:
            raise ValueError(f"This action {action} does not exist, use 1 or 0")

        reward = self.rewards[new_state]
        self.state = new_state
        return reward, new_state

    def random_action(self) -> int:
        return np.random.choice(self.action_space)

    def random_state(self) -> int:
        return np.random.choice(self.state_space)

    def reset(self, state: int) -> None:
        self.state = state

def policy_evaluation(env: MarsRover, epsilon: float, policy: List[int],
                      gamma: float):
     inf = [float("inf")]*len(env.state_space)
     value = np.array(inf)
     value_line = np.zeros(len(env.state_space))
     inf = float("inf")
     while np.linalg.norm(abs(value-value_line), ord=inf) > epsilon:
         value = value_line.copy()
         for state in env.state_space:
             env.reset(state)
             action = policy[state]
             _, new_state = env.step(action)
             value_line[state] = env.rewards[state] + gamma*value[new_state]
     return value_line

def policy_improvement(env: MarsRover, value: List[float], gamma: float,
                       policy: List[float]):
    for state in env.state_space:
        for action in env.action_space:
            env.reset(state)
            reward, new_state = env.step(action)
            q_pi = env.rewards[state] + gamma*value[new_state]
            if q_pi > value[state]:
                policy[state] = action
    return policy

def policy_iteration(env: MarsRover, epsilon: float, gamma:float):
    policy = [np.random.choice(env.action_space) for x in env.state_space]
    while True:
        value_policy = policy_evaluation(env, epsilon, policy, gamma)
        new_policy = policy_improvement(env, value_policy, gamma, policy.copy())
        if new_policy == policy:
            break
        else:
            policy = new_policy.copy()
    return value_policy, policy

rover = MarsRover(0)
#policy_iteration(rover, 0.01)

print(policy_iteration(rover, 0, 0.5))
