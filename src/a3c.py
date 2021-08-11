import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
from collections import deque

# Hyperparameters
# EPISODES = 10000
LEARNING_RATE = 0.0005
DISCOUNT_FACTOR = 0.98
T_GLOBAL_MAX = 10000
T_TREAD_MAX = 100


class Network(nn.Module):
    """
    copied from asynchronous-ppo.py
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.p = nn.Linear(128, 2)
        self.value = nn.Linear(128, 1)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob = F.softmax(self.p(x), dim=1)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)


def initilize():
    """"""


def simulate(obs, env):
    """"""
    samples, score, step = [], 0.0, 0
    while (done is not True) or (t - t_start < T_TREAD_MAX):
        # Perform a_t according to policy pi
        prob = net.pi(torch.tensor(obs).unsqueeze(0).float())
        prob_ = Categorical(prob)
        action = prob_.sample().item()

        # Receive reward r_t and new state s_t+1
        next_obs, reward, done, info = env.step(action)  # next_obs=s_t+1, reward=r_t
        samples.append((obs, action, prob[0][action], reward / 100.0, next_obs, done))
        t = t + 1
        T = T + 1

    return samples


def mini_batch(samples):
    """"""
    obs, acts, rewards, next_obs, done = zip(*samples)
    obs = torch.tensor(obs).float()
    acts = torch.tensor(acts)
    rewards = torch.tensor(rewards).float()
    next_obs = torch.tensor(next_obs).float()
    done = torch.tensor(done).int()

    return obs, acts, rewards, next_obs, done


def train(net, samples, optimizer):
    """
    개어렵넹
    """
    obs, acts, rewards, next_obs, done = mini_batch(samples)
    if done[-1] is True:
        R = 0
    else:
        R = net.v(obs[-1])  # Bootstrap from last state

    for i in reversed(range(len(samples))):
        R = rewards[i] + DISCOUNT_FACTOR * R
        p_loss = torch.log(net.pi(obs[i])) * (R - net.v(obs[i]))
        v_loss = F.mse_loss(net.v(obs[i]), R.detach())
        loss = p_loss + v_loss
        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_net.parameters(), net.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()


def predict():
    """"""


# def main():
#     """"""


if __name__ == "__main__":
    # initialize
    # until converge
    # simulate
    # mini_batch
    # train

    # initialize
    env = gym.make("CartPole-v1")
    net = Network()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    theta_global = 0
    vtheta_global = 0
    theta_thread = 0
    vtheta_thread = 0
    t = 1
    # until converge
    for t_global in range(T_GLOBAL_MAX):
        theta_thread = theta_global
        vtheta_thread = vtheta_global
        t_start = t
        if done is True:
            obs = env.reset()
            done = False
        samples = simulate(obs, env)

        # train
        train(samples, optimizer)
