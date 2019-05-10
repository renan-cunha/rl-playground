import gym
from src.q_learning import q_learning

if __name__ == "__main__":
    env = gym.make("Taxi-v2")
    num_iterations = 1000000
    learning_rate = 1
    discount_rate = 1

    q_learning(env, num_iterations, learning_rate, discount_rate, "Taxi-v2")


