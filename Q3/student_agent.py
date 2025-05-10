import gymnasium as gym
import numpy as np
import torch
from train import FeatureExtractor, Actor, make_dmc_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.env = make_dmc_env('humanoid-walk',seed=np.random.randint(0, 1000000))
        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.feature_extractor = FeatureExtractor(self.state_dim).to(device)
        self.actor = Actor(self.feature_extractor, self.act_dim).to(device)
        self.load_model("sac_icm.pth")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.feature_extractor.load_state_dict(self.actor.feature_extractor.state_dict())


    def act(self, observation):
        #return self.action_space.sample()
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(state)
            action, _ = self.actor.sample(features)
        return action.squeeze(0).cpu().numpy()
