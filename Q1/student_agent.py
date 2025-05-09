import gymnasium as gym
import numpy as np
import torch
import os
from train import DDPGAgent

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        env = gym.make("Pendulum-v1")
        self.agent = DDPGAgent(env)

        checkpoint_path = "ddpg.pth"
        if os.path.exists(checkpoint_path):
            self.agent.load(checkpoint_path)
            self.agent.actor.eval()
        else:
            raise FileNotFoundError("Checkpoint 'ddpg.pth' not found. Please train and save your model first.")

    def act(self, observation):
        #return self.action_space.sample()
        return self.agent.act(observation, add_noise=False)