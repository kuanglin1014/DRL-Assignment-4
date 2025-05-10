import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque
import gymnasium as gym
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ENV_NAME = "humanoid-walk"
SEED = 42
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA = 0.2
ALPHA_LR = 3e-4
ICM_LR = 1e-3
BATCH_SIZE = 256
REPLAY_SIZE = int(1e6)
START_STEPS = 10000
TOTAL_STEPS = 5000000
EVAL_INTERVAL = 10000
TARGET_ENTROPY = -21.0
LOG_STD_MIN = -20
LOG_STD_MAX = 2
HIDDEN_DIM = 256
SAVE_PATH = "sac_icm_checkpoint.pth"

def make_dmc_env(env_name, seed, flatten=True, use_pixels=False):
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    env = DmControltoGymnasium(env, render_mode="rgb_array", render_kwargs={"width": 256, "height": 256, "camera_id": 0})
    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    return env

class ReplayBuffer:
    def __init__(self, state_dim, act_dim, size):
        self.state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.idx = 0
        self.size = 0

    def add(self, state, act, rew, next_state, done):
        self.state_buf[self.idx] = state
        self.acts_buf[self.idx] = act
        self.rews_buf[self.idx] = rew
        self.next_state_buf[self.idx] = next_state
        self.done_buf[self.idx] = done
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return [
            torch.tensor(self.state_buf[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.acts_buf[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.rews_buf[idxs], dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(self.next_state_buf[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.done_buf[idxs], dtype=torch.float32, device=device).unsqueeze(1)
        ]

    def __len__(self):
        return self.size

class FeatureExtractor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )

    def forward(self, state):
        return self.net(state)

class Actor(nn.Module):
    def __init__(self, feature_extractor, act_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mean = nn.Linear(HIDDEN_DIM, act_dim)
        self.log_std = nn.Linear(HIDDEN_DIM, act_dim)

    def forward(self, features):
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, features):
        mean, std = self.forward(features)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return y_t, log_prob.sum(dim=-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + act_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + act_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, state, act):
        xu = torch.cat([state, act], dim=-1)
        return self.q1(xu), self.q2(xu)

    def q1_forward(self, state, act):
        return self.q1(torch.cat([state, act], dim=-1))

class ICM(nn.Module):
    def __init__(self, feature_extractor, act_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.forward_model = nn.Sequential(
            nn.Linear(HIDDEN_DIM + act_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, act_dim),
            nn.Tanh()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=ICM_LR)

    def forward(self, state, next_state, act):
        f1 = self.feature_extractor(state)
        f2 = self.feature_extractor(next_state)
        pred_f2 = self.forward_model(torch.cat([f1, act], dim=-1))
        pred_act = self.inverse_model(torch.cat([f1, f2], dim=-1))
        return pred_f2, pred_act

    def compute_intrinsic_reward(self, state, next_state, act):
        pred_f2, _ = self.forward(state, next_state, act)
        intrinsic_reward = 0.5 * F.mse_loss(pred_f2, self.feature_extractor(next_state), reduction='none').mean(dim=1)
        return intrinsic_reward

    def update(self, state, next_state, act, beta=0.2):
        f1 = self.feature_extractor(state)
        f2 = self.feature_extractor(next_state)
        pred_f2 = self.forward_model(torch.cat([f1, act], dim=-1))
        pred_act = self.inverse_model(torch.cat([f1, f2], dim=-1))

        forward_loss = F.mse_loss(pred_f2, f2)
        inverse_loss = F.mse_loss(pred_act, act)

        loss = (1 - beta) * inverse_loss + beta * forward_loss
        return loss

class SACAgent(nn.Module):
    def __init__(self, state_dim, act_dim, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.actor = Actor(self.feature_extractor, act_dim).to(device)
        self.critic = Critic(state_dim, act_dim).to(device)
        self.target_critic = Critic(state_dim, act_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.icm = ICM(self.feature_extractor, act_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=ICM_LR)
        self.log_alpha = torch.tensor(np.log(ALPHA), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)

    def update(self, replay_buffer):
        state, act, rew, next_state, done = replay_buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            intrinsic_reward = self.icm.compute_intrinsic_reward(state, next_state, act)
        total_reward = rew + intrinsic_reward

        with torch.no_grad():
            next_feat = self.feature_extractor(next_state)
            next_action, next_log_prob = self.actor.sample(next_feat)
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - torch.exp(self.log_alpha) * next_log_prob
            target_value = rew + (1 - done) * GAMMA * target_q

        current_q1, current_q2 = self.critic(state, act)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        feat = self.feature_extractor(state)
        new_action, log_prob = self.actor.sample(feat)
        q1_new = self.critic.q1_forward(state, new_action)
        actor_loss = (torch.exp(self.log_alpha) * log_prob - q1_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + TARGET_ENTROPY).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        icm_loss = self.icm.update(state, next_state, act)
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "icm": self.icm.state_dict(),
            "log_alpha": self.log_alpha,
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "icm_opt": self.icm_optimizer.state_dict(),
            "alpha_opt": self.alpha_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.icm.load_state_dict(checkpoint["icm"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_opt"])
        self.icm_optimizer.load_state_dict(checkpoint["icm_opt"])

# ========== Training ==========

def train():
    env = make_dmc_env(ENV_NAME, seed=SEED)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    feature_extractor = FeatureExtractor(state_dim).to(device)
    agent = SACAgent(state_dim, act_dim, feature_extractor)
    replay_buffer = ReplayBuffer(state_dim, act_dim, REPLAY_SIZE)

    if os.path.exists(SAVE_PATH):
        print("Loading saved model...")
        agent.load(SAVE_PATH)

    state, _ = env.reset()
    episode_return = 0

    for step in range(1, TOTAL_STEPS + 1):
        if step < START_STEPS:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                feat = agent.feature_extractor(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                act, _ = agent.actor.sample(feat)
                act = act.squeeze(0).cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

        replay_buffer.add(state, act, reward, next_state, done)
        state = next_state
        episode_return += reward

        if done:
            state, _ = env.reset()
            episode_return = 0

        if step >= START_STEPS:
            agent.update(replay_buffer)

        if step % EVAL_INTERVAL == 0:
            returns = []
            for _ in range(10):
                eval_state, _ = env.reset()
                done, total_ret = False, 0
                while not done:
                    with torch.no_grad():
                        feat = feature_extractor(torch.tensor(eval_state, dtype=torch.float32, device=device).unsqueeze(0))
                        act, _ = agent.actor.sample(feat)
                        act = act.squeeze(0).cpu().numpy()
                    eval_state, r, terminated, truncated, _ = env.step(act)
                    done = terminated or truncated
                    total_ret += r
                returns.append(total_ret)
            print(f"Step: {step}, Eval Return: {np.mean(returns):.2f}")
            agent.save(SAVE_PATH)

if __name__ == "__main__":
    train()
