import gym
from src.q_learning import q_learning

if __name__ == "__main__":
    env = gym.make("Taxi-v2")
    num_iterations = 100000
    learning_rate = 0.05  # useful for stochastic environments
    discount_rate = 0.95  # how much the agent values the next step
    exploration_rate = 1.0

    q_learning(num_iterations, learning_rate, discount_rate, exploration_rate,
               "FrozenLake-v0")


