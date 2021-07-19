import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

#Hyperparameters
EPISODES = 10000
LEARNING_RATE = 0.0005
DISCOUNT_FACTOR = 0.98
BUFFER_SIZE, START_TRAIN = 50000, 2000
BATCH_SIZE = 32

class QNet(nn.Module):
    """
    QNet
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

def minibatch_and_train(net, target_net, optimizer, buffer):
    mini_batch = random.sample(buffer, BATCH_SIZE)

    obs, acts, rewards, next_obs, done =  zip(*mini_batch)
    obs = torch.tensor(obs).float()
    acts = torch.tensor(acts)
    rewards = torch.tensor(rewards).float()
    next_obs = torch.tensor(next_obs).float()
    done = torch.tensor(done).int()

    target_q = rewards + DISCOUNT_FACTOR * done * target_net(next_obs).max(dim=1)[0]
    target_q = target_q.view(-1, 1)
    q = net(obs).gather(1, acts.view(-1, 1))
    loss = F.smooth_l1_loss(q, target_q.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net, target_net = QNet(), QNet()
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    buffer = deque(maxlen=BUFFER_SIZE)
    score, step = 0, 0
    epsilon, epsilon_decay = 0.6, 1-1e-5
    target_interval = 20
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            q_value = net(torch.tensor(obs).float())
            """e-greedy"""
            rand = random.random()
            if rand < epsilon:
                action = random.randint(0, 1)
            else:
                action = q_value.argmax().item()
            next_obs, reward, done, info = env.step(action)
            buffer.append((obs, action, reward/100.0, next_obs, done))
            obs = next_obs
            step += 1
            score += reward
            epsilon *= epsilon_decay
            
        if len(buffer) > START_TRAIN:
            minibatch_and_train(net, target_net, optimizer, buffer)
        
        if ep%target_interval==0 and ep!=0:
            target_net.load_state_dict(net.state_dict())
            
        if ep%10==0 and ep!=0:
            print('episode:{}, step:{}, avg_score:{}, len_buffer:{}, epsilon:{}'.format(ep, step, \
                  score/10.0, len(buffer), epsilon))
            score = 0
    env.close()