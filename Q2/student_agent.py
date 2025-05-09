import gymnasium
import numpy as np
import torch
import torch.nn as nn
from train import Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        obs_dim = 5
        act_dim = 1
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor.load_state_dict(torch.load("ppo_cartpole.pth", map_location=device))
        self.actor.eval()

    def act(self, observation):
        #return self.action_space.sample()
        obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, std = self.actor(obs_tensor)
            action = mean
            action_clipped = torch.clamp(action, -1.0, 1.0)
        return action_clipped.cpu().numpy()

