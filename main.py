import gym
from src.deep_q_learning import dnn_factory
from src.deep_q_learning import dqn
import matplotlib.pyplot as plt

if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    num_iterations = 500
    num_episodes = 1000
    learning_rate = 0.001  # useful for stochastic environments
    discount_rate = 0.95  # how much the agent values the next step
    exploration_rate = 1.0
    min_exploration_rate = 0.01
    exploration_rate_decay = 0.005
    num_hidden_neurons = 24
    num_hidden_layers = 1
    batch_size = 32

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    model = dnn_factory.create_dnn(input_length=state_dim,
                                   num_hidden_neurons=num_hidden_neurons,
                                   num_hidden_layers=num_hidden_layers,
                                   output_length=num_actions,
                                   learning_rate=learning_rate)

    dqn_agent = dqn.dqn(env=env, model=model,
                        exploration_rate=exploration_rate,
                        min_exploration_rate=min_exploration_rate,
                        exploration_rate_decay=exploration_rate_decay,
                        discount_factor=discount_rate)

    results1 = dqn_agent.fit(num_episodes=num_episodes,
                             num_iterations=num_iterations,
                             batch_size=batch_size)

    dqn_agent = dqn.dqn(env=env, model=model,
                        exploration_rate=exploration_rate,
                        min_exploration_rate=min_exploration_rate,
                        exploration_rate_decay=0.001,
                        discount_factor=discount_rate)

    results2 = dqn_agent.fit(num_episodes, num_iterations, batch_size)

    plt.plot(results1, label="0.005")
    plt.plot(results2, label="0.001")
    plt.legend()
    plt.show()



