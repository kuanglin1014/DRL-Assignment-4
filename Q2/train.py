import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
LR = 3e-4
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_UPDATE = 2048
TOTAL_TIMESTEPS = 2000000

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

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        hidden = self.net(x)
        mean = self.policy_mean(hidden)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# Sample from a Gaussian distribution
def sample_action(mean, std):
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(-1)
    return action, log_prob

# Compute advantage estimates
def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    adv = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    return adv

# Main training loop
def train():
    env = make_dmc_env("cartpole-balance", seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR)

    obs, _ = env.reset()
    episode_rewards = deque(maxlen=10)
    ep_reward = 0
    global_step = 0

    while global_step < TOTAL_TIMESTEPS:
        # Storage for rollout
        observations, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        for _ in range(STEPS_PER_UPDATE):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            mean, std = actor(obs_tensor)
            value = critic(obs_tensor)
            action, log_prob = sample_action(mean, std)

            action_env = action.cpu().detach().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated

            observations.append(obs)
            actions.append(action_env)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            obs = next_obs
            ep_reward += reward
            global_step += 1

            if done:
                obs, _ = env.reset()
                episode_rewards.append(ep_reward)
                ep_reward = 0

        # Final bootstrap value
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        next_value = critic(obs_tensor).item()
        values.append(next_value)

        # Convert to torch tensors
        values = np.array(values)
        rewards = np.array(rewards)
        dones = np.array(dones, dtype=np.float32)
        advantages = compute_gae(rewards, values, dones)
        returns = advantages + values[:-1]

        # Normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training
        dataset = list(zip(observations, actions, log_probs, returns, advantages))
        for _ in range(EPOCHS):
            np.random.shuffle(dataset)
            for i in range(0, len(dataset), BATCH_SIZE):
                batch = dataset[i:i+BATCH_SIZE]
                obs_b, act_b, old_log_b, ret_b, adv_b = zip(*batch)

                obs_b = torch.tensor(np.array(obs_b), dtype=torch.float32).to(device)
                act_b = torch.tensor(np.array(act_b), dtype=torch.float32).to(device)
                old_log_b = torch.tensor(np.array(old_log_b), dtype=torch.float32).to(device)
                ret_b = torch.tensor(np.array(ret_b), dtype=torch.float32).to(device)
                adv_b = torch.tensor(np.array(adv_b), dtype=torch.float32).to(device)

                mean, std = actor(obs_b)
                dist = torch.distributions.Normal(mean, std)
                log_probs_new = dist.log_prob(act_b).sum(-1)
                entropy = dist.entropy().sum(-1)

                ratio = torch.exp(log_probs_new - old_log_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value = critic(obs_b)
                value_loss = (ret_b - value).pow(2).mean()

                loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy.mean()

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()

        if global_step % 10000 < STEPS_PER_UPDATE:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"Step: {global_step}, Avg Reward: {avg_reward:.2f}")
            torch.save(actor.state_dict(), "ppo_cartpole.pth")

    torch.save(actor.state_dict(), "ppo_cartpole.pth")
    print("Model saved to ppo_cartpole.pth")

if __name__ == "__main__":
    train()
