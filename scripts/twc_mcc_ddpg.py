import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import json
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from ddpg import DDPGEngine
from utils import ReplayBuffer, OUNoise
from mlp import Critic
from twc import (
    build_twc,
    mcc_obs_encoder,
    mcc_obs_encoder_speed_weighted,
    twc_out_2_mcc_action,
    twc_out_2_mcc_action_tanh,
)

# Algorithm and hyperparams based on
# https://arxiv.org/pdf/1509.02971
# --- Environment ---
ENV = "MountainCarContinuous-v0"
SEED = 42
# --- Hyperparameters ---
MAX_EPISODE        = 500
MAX_TIME_STEPS     = 999
WARMUP_STEPS       = 10_000
BATCH_SIZE         = 128
NUM_UPDATE_LOOPS   = 1
UPDATE_EVERY       = 2
GAMMA              = 0.99
TAU                = 0.005
ACTOR_LR           = 5e-3
CRITIC_LR          = 5e-4
MINI_BATCH         = 3
WORST_K            = 1
TWC_INTERNAL_STEPS = 3
CRITIC_HID_LAYERS  = [400, 300]
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA_START, SIGMA_END, SIGMA_DECAY_EPIS = 0.20, 0.05, 200

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
    'mini_batch': MINI_BATCH,
    'worst_k': WORST_K,
    'critic_layers': CRITIC_HID_LAYERS,
    'twc_internal_steps': TWC_INTERNAL_STEPS,
    'env': ENV,
    'seed': SEED,
    'sigma_start': SIGMA_START,
    'sigma_end': SIGMA_END,
    'sigma_ep_end': SIGMA_DECAY_EPIS,
    'device': str(DEVICE),
}

# --- HELPER FUNCTIONS ---
@torch.no_grad()
def evaluate_policy(env, ddpg: DDPGEngine, episodes=10):
    ddpg.actor.eval()
    total = 0.0
    eval_actions = []
    for _ in range(episodes):
        obs, _ = env.reset()
        ddpg.actor.reset()
        done = False
        while not done:
            a = ddpg.get_action(obs, action_noise=None)
            eval_actions.append(a)
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
    ddpg.actor.train()
    return (total / episodes), np.mean(eval_actions)
    
# -------------------------

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
env = gym.make(ENV)
env.reset(seed=SEED)
env.action_space.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = build_twc(obs_encoder=mcc_obs_encoder_speed_weighted,
                  action_decoder=twc_out_2_mcc_action_tanh,
                  internal_steps=TWC_INTERNAL_STEPS,
                  log_stats=False)

critic = Critic(state_dim, action_dim, size=CRITIC_HID_LAYERS)


replay_buf = ReplayBuffer(obs_dim=env.observation_space.shape[0],
                          act_dim=env.action_space.shape[0],
                          size=100_000,
                          keep=WARMUP_STEPS)

ou_noise = OUNoise(action_dimension=env.action_space.shape[0],
                   mu=0,
                   theta=0.15,
                   sigma=SIGMA_START,
                   sigma_end=SIGMA_END,
                   sigma_decay_epis=SIGMA_DECAY_EPIS)

actor_opt = torch.optim.Adam(actor.parameters(),  lr=ACTOR_LR)
critic_opt= torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

ddpg = DDPGEngine(gamma=GAMMA,
                  tau=TAU,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  actor=actor,
                  critic=critic,
                  actor_optimizer=actor_opt,
                  critic_optimizer=critic_opt,
                  update_every=UPDATE_EVERY,
                  device=DEVICE)


obs, _ = env.reset()
ep_ret, ep_len = 0, 0
best_ret = -np.inf
total_steps = 0
# --- TRAIN LOOP ---
# Create a unique run directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f"twc_ddpg_{timestamp}"
writer = SummaryWriter(f'runs/ddpg/{run_name}')

os.makedirs(writer.log_dir, exist_ok=True)
params_path = os.path.join(writer.log_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(params, f, indent=4)

for e in tqdm(range(MAX_EPISODE)):
    obs, _ = env.reset()
    ou_noise.update_sigma()
    ou_noise.reset()    
    ep_reward = 0
    steps = 0
    ddpg.actor.reset()
    ep_actions = []
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
            ep_actions.append(action)
            for _ in range(NUM_UPDATE_LOOPS):
                batch = replay_buf.sample(BATCH_SIZE, DEVICE)
                #actor_loss, critic_loss = ddpg.update_step(batch)
                actor_loss, critic_loss = ddpg.update_step_worst_k(replay=replay_buf,
                                                                   batch_size=BATCH_SIZE, 
                                                                   M=MINI_BATCH,
                                                                   WORST=WORST_K
                                                                   )
                # Log losses
                writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                writer.add_scalar('Loss/Critic', critic_loss, total_steps)

        if done:
            break

    # Log episode return
    writer.add_scalar('Training/Episode_Return', ep_reward, e)
    writer.add_scalar('Training/Episode_steps', steps, e)
    writer.add_scalar('Training/AvgAction', np.mean(ep_actions), e)

    if (e+1) % 10 == 0:
        eval_ret, eval_avg_action = evaluate_policy(env, ddpg, episodes=10)
        # Log evaluation return
        writer.add_scalar('Evaluation/Return', eval_ret, e)
        writer.add_scalar('Evaluation/AvgAction', eval_avg_action, e)
        #writer.add_scalar('Evaluation/Return_steps', eval_ret, total_steps)
        print(f"Evaluation after Episode {e+1}: {eval_ret:.2f}")
        if eval_ret > best_ret:
            best_ret = eval_ret
            torch.save(ddpg.actor.state_dict(), f"models/twc_ddpg_actor_best_{timestamp}.pth")
            torch.save(ddpg.critic.state_dict(), f"models/twc_ddpg_critic_best_{timestamp}.pth")
            print(f"New best evaluation reward: {best_ret:.2f}, models saved.")

print("Training completed.")

env.close()
# save final models
torch.save(ddpg.actor.state_dict(), f"models/twc_ddpg_final_{timestamp}.pth")
torch.save(ddpg.critic.state_dict(), f"models/twc_ddpg_critic_final_{timestamp}.pth")

# Close tensorboard writer
writer.close()
