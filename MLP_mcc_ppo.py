import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PPO_engine.ppo_engine import PPOEngine
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Tuple, List


class PPOActor(nn.Module):
    """
    PPO Actor network for continuous action spaces.
    Outputs mean and log_std for a Gaussian policy.
    """
    def __init__(self, state_dim: int, action_dim: int, sizes: List[int], max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        
        # Shared layers
        self.shared_net = nn.Sequential(
            nn.LayerNorm(normalized_shape=state_dim),
            nn.Linear(state_dim, sizes[0]),
            nn.ReLU(),
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU()
        )
        
        # Mean head
        self.mean_head = nn.Linear(sizes[1], action_dim)
        
        # Log std head (learnable parameter)
        self.log_std_head = nn.Linear(sizes[1], action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Input state tensor
            
        Returns:
            mean: Mean of the action distribution
            log_std: Log standard deviation of the action distribution
        """
        shared_features = self.shared_net(state)
        
        mean = self.mean_head(shared_features)
        log_std = self.log_std_head(shared_features)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        
        # Scale mean to action range
        mean = torch.tanh(mean) * self.max_action
        
        return mean, log_std


class PPOCritic(nn.Module):
    """
    PPO Critic network for value function estimation.
    """
    def __init__(self, state_dim: int, sizes: List[int]):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape=state_dim),
            nn.Linear(state_dim, sizes[0]),
            nn.ReLU(),
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Input state tensor
            
        Returns:
            value: State value estimate
        """
        return self.net(state)


# --- Environment ---
ENV = "MountainCarContinuous-v0"
SEED = 42

# --- PPO Hyperparameters ---
MAX_EPISODES = 300
MAX_TIMESTEPS_PER_EPISODE = 1000
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 64

# PPO specific parameters
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_RATIO = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.3
MAX_GRAD_NORM = 0.5

# Network parameters
ACTOR_HIDDEN_SIZES = [64, 64]
CRITIC_HIDDEN_SIZES = [64, 64]
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

# Evaluation parameters
EVAL_FREQUENCY = 10
EVAL_EPISODES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_policy(env: gym.Env, ppo_engine: PPOEngine, episodes: int = 10) -> float:
    """
    Evaluate the current policy.
    
    Args:
        env: Environment to evaluate on
        ppo_engine: PPO engine with trained policy
        episodes: Number of episodes to evaluate
        
    Returns:
        Average return over evaluation episodes
    """
    ppo_engine.set_eval()
    total_returns = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            action, _, _ = ppo_engine.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
            
        total_returns.append(episode_return)
    
    ppo_engine.set_train()
    return np.mean(total_returns)


def collect_episode_data(env: gym.Env, ppo_engine: PPOEngine, max_steps: int) -> Tuple[List, List, List, List, List, List]:
    """
    Collect one episode of data for PPO training.
    
    Args:
        env: Environment to interact with
        ppo_engine: PPO engine for action selection
        max_steps: Maximum steps per episode
        
    Returns:
        Tuple of (states, actions, log_probs, rewards, values, dones)
    """
    obs, _ = env.reset()
    
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    
    for _ in range(max_steps):
        # Get action from current policy
        action, log_prob, value = ppo_engine.get_action(obs)
        
        # Store data
        states.append(obs.copy())
        actions.append(action.copy())
        log_probs.append(log_prob[0])
        values.append(value[0])
        
        # Take action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        dones.append(done)
        
        obs = next_obs
        
        if done:
            break
    
    return states, actions, log_probs, rewards, values, dones


def main():
    """Main training function."""
    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Create environment
    env = gym.make(ENV)
    env.action_space.seed(SEED)
    
    # Create networks
    actor = PPOActor(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        sizes=ACTOR_HIDDEN_SIZES,
        max_action=env.action_space.high[0]
    )
    
    critic = PPOCritic(
        state_dim=env.observation_space.shape[0],
        sizes=CRITIC_HIDDEN_SIZES
    )
    
    # Create optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)
    
    # Create PPO engine
    ppo_engine = PPOEngine(
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
        device=DEVICE
    )
    
    # Create TensorBoard writer
    run_name = f"mlp_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f'runs/{run_name}')
    
    # Training variables
    best_eval_return = -np.inf
    total_timesteps = 0
    
    print(f"Starting PPO training on {ENV}")
    print(f"Device: {DEVICE}")
    print(f"Max episodes: {MAX_EPISODES}")
    
    # Training loop
    for episode in tqdm(range(MAX_EPISODES), desc="Training"):
        # Collect episode data
        states, actions, log_probs, rewards, values, dones = collect_episode_data(
            env, ppo_engine, MAX_TIMESTEPS_PER_EPISODE
        )
        
        # Get next state value for bootstrap
        if len(states) > 0:
            next_state = torch.as_tensor(states[-1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                next_value = ppo_engine.critic(next_state).item()
        else:
            next_value = 0.0
        
        # Update PPO policy
        if len(states) >= MINIBATCH_SIZE:
            losses = ppo_engine.update_step(
                states=states,
                actions=actions,
                old_log_probs=log_probs,
                rewards=rewards,
                values=values,
                dones=dones,
                next_value=next_value,
                epochs=UPDATE_EPOCHS
            )
            
            # Log training metrics
            episode_return = sum(rewards)
            episode_length = len(rewards)
            total_timesteps += episode_length
            
            writer.add_scalar('Training/Episode_Return', episode_return, episode)
            writer.add_scalar('Training/Episode_Length', episode_length, episode)
            writer.add_scalar('Training/Actor_Loss', losses['actor_loss'], episode)
            writer.add_scalar('Training/Critic_Loss', losses['critic_loss'], episode)
            writer.add_scalar('Training/Entropy', losses['entropy'], episode)
            writer.add_scalar('Training/Advantage_Mean', losses['advantage_mean'], episode)
            writer.add_scalar('Training/Advantage_Std', losses['advantage_std'], episode)
        
        # Evaluation
        if (episode + 1) % EVAL_FREQUENCY == 0:
            eval_return = evaluate_policy(env, ppo_engine, EVAL_EPISODES)
            writer.add_scalar('Evaluation/Return', eval_return, episode)
            
            print(f"Episode {episode + 1}: Eval Return = {eval_return:.2f}")
            
            # Save best model
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                torch.save(ppo_engine.get_save_bundle(), f"models/ppo_mcc_best.pth")
                print(f"New best evaluation return: {best_eval_return:.2f}")
    
    # Save final model
    torch.save(ppo_engine.get_save_bundle(), f"models/ppo_mcc_final.pth")
    
    print(f"Training completed!")
    print(f"Best evaluation return: {best_eval_return:.2f}")
    
    # Cleanup
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
