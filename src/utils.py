import torch


def list_to_torch(mini_batch):
    obs, acts, rewards, next_obs, done = zip(*mini_batch)
    obs = torch.tensor(obs).float()
    acts = torch.tensor(acts)
    rewards = torch.tensor(rewards).float()
    next_obs = torch.tensor(next_obs).float()
    done = torch.tensor(done).int()
