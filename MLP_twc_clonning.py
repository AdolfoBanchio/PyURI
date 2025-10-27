import os
import json
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DDPG_engine.ddpg_engine import DDPGEngine
from DDPG_engine.replay_buffer import ReplayBuffer
from utils.MLP_models import Critic
from utils.twc_builder import build_twc
from utils.twc_io import twc_out_2_mcc_action, mcc_obs_encoder, twc_out_2_mcc_action_tanh
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Algorithm and hyperparams based on
# https://arxiv.org/pdf/1509.02971
# --- Environment ---
ENV = "MountainCarContinuous-v0"
SEED = 42
# --- Hyperparameters ---
MAX_EPISODE = 500
MAX_TIME_STEPS = 999
WARMUP_STEPS = 10_000
BATCH_SIZE        = 64
NUM_UPDATE_LOOPS  = 1
UPDATE_EVERY = 2
GAMMA             = 0.99
TAU               = 0.001
ACTOR_LR = 1e-5
CRITIC_LR= 1e-3

SIGMA_START, SIGMA_END, SIGMA_DECAY_EPIS = 0.20, 0.05, 150

CRITIC_HID_LAYERS= [20, 10]
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dict to save hyerparams
params = {
    'max_episode': MAX_EPISODE,
    'max_time_steps': MAX_TIME_STEPS,
    'warmup_steps': WARMUP_STEPS,
    'batch_size': BATCH_SIZE,
    'num_update_loops': NUM_UPDATE_LOOPS,
    'update_every': UPDATE_EVERY,
    'gamma': GAMMA,
    'tau': TAU,
    'actor_lr': ACTOR_LR,
    'critic_lr': CRITIC_LR,
    'critic_layers': CRITIC_HID_LAYERS,
    'env': ENV,
    'seed': SEED,
    'sigma_start': SIGMA_START,
    'sigma_end': SIGMA_END,
    'sigma_ep_end': SIGMA_DECAY_EPIS,
    'device': str(DEVICE),
}
# --- HELPER FUNCTIONS ---
def sigma_for_episode(ep, start=SIGMA_START, end=SIGMA_END, decay_episodes=SIGMA_DECAY_EPIS):
    frac = min(1.0, ep / decay_episodes)
    return start + (end - start) * frac

@torch.no_grad()
def evaluate_policy(env, ddpg: DDPGEngine, episodes=10):
    ddpg.actor.eval()
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        ddpg.actor.reset()
        done = False
        while not done:
            a = ddpg.get_action(obs, action_noise=None)
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
    ddpg.actor.train()
    return total / episodes

class OUNoise:
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
# -------------------------

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
env = gym.make(ENV)
env.reset(seed=SEED)
env.action_space.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = build_twc(obs_encoder=mcc_obs_encoder,
                  action_decoder=twc_out_2_mcc_action,
                  log_stats=False)

critic = Critic(state_dim, action_dim, size=CRITIC_HID_LAYERS)

critic_path = "models/ddpg_critic_best.pth"
state_dict = torch.load(critic_path, map_location=DEVICE)
critic.load_state_dict(state_dict)
critic.to(DEVICE)
critic.eval()
for param in critic.parameters():
    param.requires_grad_(False)
    
replay_buf = ReplayBuffer(obs_dim=env.observation_space.shape[0],
                          act_dim=env.action_space.shape[0],
                          size=100_000,
                          keep=WARMUP_STEPS)

ou_noise = OUNoise(action_dimension=env.action_space.shape[0])

actor_opt = torch.optim.Adam(actor.parameters(),  lr=ACTOR_LR)

ddpg = DDPGEngine(gamma=GAMMA,
                  tau=TAU,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  actor=actor,
                  critic=critic,
                  actor_optimizer=actor_opt,
                  critic_optimizer=None,
                  update_every=UPDATE_EVERY,
                  device=DEVICE)


obs, _ = env.reset()
ep_ret, ep_len = 0, 0
best_ret = -np.inf
total_steps = 0
# --- TRAIN LOOP ---
# Create a unique run directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f"twc_ddpg_clonning_{timestamp}"
writer = SummaryWriter(f'runs/{run_name}')

os.makedirs(writer.log_dir, exist_ok=True)
params_path = os.path.join(writer.log_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(params, f, indent=4)

for e in tqdm(range(MAX_EPISODE)):
    obs, _ = env.reset()
    ou_noise.sigma = sigma_for_episode(e)
    ou_noise.reset()    
    ep_reward = 0
    steps = 0
    ddpg.actor.reset()

    for t in range(MAX_TIME_STEPS):
        if total_steps < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            action = ddpg.get_action(obs, action_noise=ou_noise.noise)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1
        
        replay_buf.store(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs
        total_steps += 1

        if total_steps > WARMUP_STEPS and replay_buf.size >= BATCH_SIZE:
            for _ in range(NUM_UPDATE_LOOPS):
                batch = replay_buf.sample(BATCH_SIZE, DEVICE)
                actor_loss, critic_loss = ddpg.update_step_with_Qexpert(batch)
                # Log losses
                writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                writer.add_scalar('Loss/Critic', critic_loss, total_steps)

        if done:
            break

    # Log episode return
    writer.add_scalar('Training/Episode_Return', ep_reward, e)
    writer.add_scalar('Training/Episode_steps', steps, e)

    if (e+1) % 10 == 0:
        eval_ret = evaluate_policy(env, ddpg, episodes=10)
        # Log evaluation return
        writer.add_scalar('Evaluation/Return', eval_ret, e)
        writer.add_scalar('Evaluation/Return_steps', eval_ret, total_steps)
        print(f"Evaluation after Episode {e+1}: {eval_ret:.2f}")
        if eval_ret > best_ret:
            best_ret = eval_ret
            torch.save(ddpg.actor.state_dict(), f"models/twc_ddpg_actor_clone_best_{timestamp}.pth")
            print(f"New best evaluation reward: {best_ret:.2f}, models saved.")

print("Training completed.")

env.close()
# save final models
torch.save(ddpg.actor.state_dict(), f"models/ddpg_twc_final_clone_{timestamp}.pth")

# Close tensorboard writer
writer.close()
