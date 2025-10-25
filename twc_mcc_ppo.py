import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PPO_engine.ppo_engine import PPOEngine
from utils.ppo_models import PPOCritic
from utils.twc_builder import build_twc
from utils.twc_io import mcc_obs_encoder, twc_out_2_mcc_drive
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# --- Environment ---
ENV = "MountainCarContinuous-v0"
SEED = 42

# --- Hyperparameters ---
EPOCHS = 1000
STEPS_PER_EPOCH = 2048
TRAIN_EPOCHS = 10
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_RATIO = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

# Initialize environment
env = gym.make(ENV)
env.action_space.seed(SEED)

# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build TWC actor with proper drive decoder for PPO
class TWCActorPPO(nn.Module):
    def __init__(self, twc_model, init_log_std=-0.5):
        super().__init__()
        self.twc = twc_model
        self.log_std = nn.Parameter(torch.ones(1) * init_log_std)
    
    def forward(self, state):
        mean = self.twc(state)  # Uses twc_out_2_mcc_drive internally
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

# Build networks
base_twc = build_twc(
    obs_encoder=mcc_obs_encoder,
    action_decoder=twc_out_2_mcc_drive,
    log_stats=False
)
actor = TWCActorPPO(base_twc).to(device)
critic = PPOCritic(state_dim=2).to(device)  # MCC has 2D state space

# Initialize optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

# Initialize PPO engine
ppo = PPOEngine(
    gamma=GAMMA,
    lambda_gae=LAMBDA_GAE,
    clip_ratio=CLIP_RATIO,
    value_loss_coef=VALUE_LOSS_COEF,
    entropy_coef=ENTROPY_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    observation_space=env.observation_space,
    action_space=env.action_space,
    actor=actor,
    critic=critic,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    device=device
)

# Initialize tensorboard
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/twc_ppo_{timestamp}")

# Training loop
best_reward = float('-inf')
state, _ = env.reset(seed=SEED)

for epoch in tqdm(range(EPOCHS)):
    # Collect trajectory
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    
    # Collect steps for this epoch
    for step in range(STEPS_PER_EPOCH):
        # Get action from policy
        action, log_prob, value = ppo.get_action(state)
        
        # Take step in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store step data
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(done)
        
        # Update state
        state = next_state if not done else env.reset(seed=SEED)[0]
        
    # Get value estimate for final state
    _, _, next_value = ppo.get_action(state)
    
    # Update policy
    metrics = ppo.update_step(
        states=states,
        actions=actions,
        old_log_probs=log_probs,
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        epochs=TRAIN_EPOCHS
    )
    
    # Calculate episode statistics
    episode_reward = sum(rewards)
    
    # Log to tensorboard
    writer.add_scalar('Reward/train', episode_reward, epoch)
    writer.add_scalar('Loss/actor', metrics['actor_loss'], epoch)
    writer.add_scalar('Loss/critic', metrics['critic_loss'], epoch)
    writer.add_scalar('Loss/entropy', metrics['entropy'], epoch)
    
    # Save best model
    if episode_reward > best_reward:
        best_reward = episode_reward
        actor_sate_dict = ppo.get_save_bundle()['actor']
        torch.save(actor_sate_dict, f'models/twc_ppo_best.pth')
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Actor Loss: {metrics['actor_loss']:.4f}")
        print(f"Critic Loss: {metrics['critic_loss']:.4f}")
        print(f"Entropy: {metrics['entropy']:.4f}\n")

# Close environment and writer
env.close()
writer.close()