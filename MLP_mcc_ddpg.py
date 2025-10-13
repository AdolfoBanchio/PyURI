import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DDPG_engine.ddpg_engine import DDPGEngine
from DDPG_engine.replay_buffer import ReplayBuffer
from utils.MLP_models import Actor, Critic
from tqdm import tqdm

HIDDEN_SIZE = [64, 64] # OPEN AI
# --- Environment ---
ENV = "MountainCarContinuous-v0"
SEED = 42
# --- Hyperparameters ---
TOTAL_STEPS       = 100_000
SAVE_EVERY        = TOTAL_STEPS // 4
UPDATE_AFTER      = 5_000
UPDATE_EVERY      = 1
NUM_UPDATE_LOOPS  = 1
EVAL_INTERVAL     = 5_000
NOISE_STD_INIT    = 0.35
NOISE_STD_FINAL   = 0.1
NOISE_DECAY_STEPS = 70_000
GAMMA             = 0.99
TAU               = 0.005
BATCH_SIZE        = 128
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
env = gym.make(ENV)
env.action_space.seed(SEED)

actor = Actor(hidden_size=HIDDEN_SIZE,
              num_inputs=env.observation_space.shape[0],
              action_space=env.action_space)
critic = Critic(hidden_size=HIDDEN_SIZE,
                num_inputs=env.observation_space.shape[0],
                action_space=env.action_space)

replay_buf = ReplayBuffer(obs_dim=env.observation_space.shape[0],
                          act_dim=env.action_space.shape[0])

actor_opt = torch.optim.Adam(actor.parameters(),  lr=1e-4)
critic_opt= torch.optim.Adam(critic.parameters(), lr=5e-4)

ddpg = DDPGEngine(gamma=GAMMA,
                  tau=TAU,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  actor=actor,
                  critic=critic,
                  actor_optimizer=actor_opt,
                  critic_optimizer=critic_opt,
                  device=DEVICE)


obs, _ = env.reset()
ep_ret, ep_len = 0, 0
best_ret = -np.inf


def current_noise_std(t):
    frac = np.clip(t / NOISE_DECAY_STEPS, 0., 1.)
    return NOISE_STD_INIT + frac * (NOISE_STD_FINAL - NOISE_STD_INIT)

@torch.no_grad()
def evaluate_policy(env, actor, device, episodes=5):
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = ddpg.get_action(obs_t)
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
    return total / episodes

for t in tqdm(range(TOTAL_STEPS)):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    action = ddpg.get_action(obs_t, action_noise=lambda: np.random.normal(0, current_noise_std(t), size=env.action_space.shape[0]))

    next_obs, r, ter, trunc, _ = env.step(action)
    done = ter or trunc
    ep_ret += r
    ep_len += 1
    replay_buf.store(obs, action, r, next_obs, done)
    obs = next_obs

    if t>=UPDATE_AFTER and t % UPDATE_EVERY == 0:
        for _ in range(NUM_UPDATE_LOOPS):
            batch = replay_buf.sample(BATCH_SIZE, DEVICE)
            ddpg.update_step(batch)

    if done:
        obs, _ = env.reset()
        if ep_ret > best_ret:
            best_ret = ep_ret
        print(f"Episode Return: {ep_ret:.2f} \t Episode Length: {ep_len} \t Best Return: {best_ret:.2f}")
        ep_ret, ep_len = 0, 0
    
    if (t+1) % EVAL_INTERVAL == 0:
        eval_ret = evaluate_policy(env, ddpg.actor, DEVICE)
        print(f"Evaluation Return: {eval_ret:.2f}")

    if (t+1) % SAVE_EVERY == 0:
        torch.save(ddpg.actor.state_dict(), f"ddpg_actor_{t+1}.pth")
        torch.save(ddpg.critic.state_dict(), f"ddpg_critic_{t+1}.pth")


print("Training completed.")
env.close()

# save final models
torch.save(ddpg.actor.state_dict(), f"ddpg_actor_final.pth")
torch.save(ddpg.critic.state_dict(), f"ddpg_critic_final.pth")
