import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import gym
import numpy as np

env = gym.make("CartPole-v0")

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(64, env.action_space.n)
        self.out_act = nn.Softmax()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        a2 = self.fc2(h1)
        h2 = self.relu1(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

    

def select_action(state: np.ndarray, policy: Net) -> int:
    output = policy.forward(state)
    return np.random.choide(env.action_space, p=output)


net = Net()
opt = optim.Adam(net.parameters(), lr=0.001)
#riterion = nn.BCELoss()

state = env.reset()
num_iterations = 100000
transitions = []
gamma = 0.95
for i in range(num_iterations):
    action = select_action(state, net)
    next_state, reward, done, _ = env.step(action)
    transitions.append((state, reward, next_state))
    if done:
        g = sum([transisitions[x][1]*0.95**x for x in range(len(transitions))])
        state = env.reset()
        transitions = []
    else:
        state = next_state

