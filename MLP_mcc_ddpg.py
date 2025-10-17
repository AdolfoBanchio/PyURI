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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Algorithm and hyperparams based on
# https://arxiv.org/pdf/1509.02971
# --- Environment ---
ENV = "MountainCarContinuous-v0"
SEED = 42
# --- Hyperparameters ---
MAX_EPISODE = 220
NUM_UPDATE_LOOPS  = 1
WARMUP_STEPS = 3000
# _________________________
GAMMA             = 0.98
TAU               = 0.00034
BATCH_SIZE        = 170
MAX_TIME_STEPS = 1000
# _________________________
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTOR_HID_LAYERS = [64, 128]
CRITIC_HID_LAYERS= [64, 128]
CRITIC_LR= 0.00346
ACTOR_LR = 0.00032

# --- HELPER FUNCTIONS ---
@torch.no_grad()
def evaluate_policy(env, ddpg: DDPGEngine, episodes=10):
    ddpg.actor.eval()
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a = ddpg.get_action(obs, action_noise=None)
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
    ddpg.actor.train()
    return total / episodes

class OUNoise:
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.184):
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
env.action_space.seed(SEED)

actor = Actor(state_dim=env.observation_space.shape[0],
              action_dim=env.action_space.shape[0],
              sizes=ACTOR_HID_LAYERS)

critic = Critic(state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                sizes=CRITIC_HID_LAYERS)

replay_buf = ReplayBuffer(obs_dim=env.observation_space.shape[0],
                          act_dim=env.action_space.shape[0],
                          size=52_000)

ou_noise = OUNoise(action_dimension=env.action_space.shape[0])

actor_opt = torch.optim.Adam(actor.parameters(),  lr=ACTOR_LR)
critic_opt= torch.optim.Adam(critic.parameters(), lr=CRITIC_LR, weight_decay=1e-2)

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
total_steps = 0
# --- TRAIN LOOP ---
# Create a unique run directory with timestamp
run_name = f"mlp_ddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(f'runs/{run_name}')

for e in tqdm(range(MAX_EPISODE)):
    obs, _ = env.reset()
    ou_noise.reset()
    ep_reward = 0
    
    steps = 0
    for t in range(MAX_TIME_STEPS):
        action = ddpg.get_action(
            obs, 
            action_noise=ou_noise.noise
        )

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1
        
        replay_buf.store(obs, action, reward, next_obs, done)
        obs = next_obs
        total_steps += 1

        if total_steps > WARMUP_STEPS and replay_buf.size >= BATCH_SIZE:
            for _ in range(NUM_UPDATE_LOOPS):
                batch = replay_buf.sample(BATCH_SIZE, DEVICE)
                actor_loss, critic_loss = ddpg.update_step(batch)
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
        print(f"Evaluation after Episode {e+1}: {eval_ret:.2f}")
        if eval_ret > best_ret:
            best_ret = eval_ret
            torch.save(ddpg.actor.state_dict(), f"models/ddpg_actor_best.pth")
            torch.save(ddpg.critic.state_dict(), f"models/ddpg_critic_best.pth")
            print(f"New best evaluation reward: {best_ret:.2f}, models saved.")

print("Training completed.")

env.close()
# save final models
torch.save(ddpg.actor.state_dict(), f"models/ddpg_actor_final.pth")
torch.save(ddpg.critic.state_dict(), f"models/ddpg_critic_final.pth")

# Close tensorboard writer
writer.close()