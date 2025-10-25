import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional


class PPOEngine:
    """
    Proximal Policy Optimization (PPO) engine for training actor-critic models.
    
    This class provides a complete PPO implementation with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function updates
    - Policy entropy regularization
    
    Parameters:
        - gamma: float, discount factor for future rewards
        - lambda_gae: float, GAE parameter for advantage estimation
        - clip_ratio: float, PPO clipping parameter (typically 0.1-0.3)
        - value_loss_coef: float, coefficient for value function loss
        - entropy_coef: float, coefficient for entropy regularization
        - max_grad_norm: float, maximum gradient norm for clipping
        - observation_space: gym.Space, observation space
        - action_space: gym.Space, action space
        - actor: nn.Module, policy network (should output action logits)
        - critic: nn.Module, value function network
        - actor_optimizer: torch.optim.Optimizer, optimizer for actor
        - critic_optimizer: torch.optim.Optimizer, optimizer for critic
        - device: torch.device, device to run computations on
    """
    
    def __init__(self,
                 gamma: float,
                 lambda_gae: float,
                 clip_ratio: float,
                 value_loss_coef: float,
                 entropy_coef: float,
                 max_grad_norm: float,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 actor: nn.Module,
                 critic: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer,
                 device: torch.device):
        
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.obs_space = observation_space
        self.act_space = action_space
        
        self.actor = actor
        self.critic = critic
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.device = device
        
        # Move networks to device
        self.actor.to(device)
        self.critic.to(device)
        
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get action from the current policy.
        
        Args:
            state: Current state
            deterministic: If True, return deterministic action (mean for continuous, argmax for discrete)
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: State value estimate
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            # Get action distribution
            if hasattr(self.act_space, 'n'):  # Discrete action space
                logits = self.actor(state)
                dist = torch.distributions.Categorical(logits=logits)
                
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = dist.sample()
                    
                log_prob = dist.log_prob(action)
                action = action.cpu().numpy()
                
            else:  # Continuous action space
                mean, log_std = self.actor(state)
                std = torch.exp(log_std)
                base_dist = torch.distributions.Normal(mean, std)
                
                if deterministic:
                    z = mean
                else:
                    z = base_dist.sample()
                
                a = torch.tanh(z) # goes to -1, 1

                action = a

                log_prob = base_dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
                # Clip action to action space bounds
            # Get value estimate
            value = self.critic(state)
            value = value.cpu().numpy()
            
        self.actor.train()
        self.critic.train()
        
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value.flatten()
    
    def compute_gae(self, rewards: List[float], 
                   values: List[float], 
                   dones: List[bool],
                   next_value: float) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for the next state
            
        Returns:
            advantages: List of advantage estimates
            returns: List of return estimates
        """
        advantages = []
        returns = []
        
        # Add next value for bootstrap
        values = values + [next_value]
        
        # Compute advantages and returns backwards
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
            
        return advantages, returns
    
    def update_step(self, 
                   states: List[np.ndarray],
                   actions: List[np.ndarray], 
                   old_log_probs: List[float],
                   rewards: List[float],
                   values: List[float],
                   dones: List[bool],
                   next_value: float,
                   epochs: int = 4) -> Dict[str, float]:
        """
        Perform PPO update step.
        
        Args:
            states: List of states
            actions: List of actions taken
            old_log_probs: List of log probabilities from old policy
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
            epochs: Number of epochs to train
            
        Returns:
            Dictionary with loss values
        """
        # Convert to tensors
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.array(old_log_probs), dtype=torch.float32, device=self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = torch.as_tensor(np.array(advantages), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(np.array(returns), dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Some actor implementations keep recurrent state inside the module.
        # Ensure those cached tensors do not carry autograd history across
        # optimization epochs, otherwise PyTorch will see them as a second
        # backward through the same graph.
        detach_state_fn = None
        actor_level_detach = getattr(self.actor, "detach", None)
        if callable(actor_level_detach):
            detach_state_fn = actor_level_detach
        else:
            stateful_submodule = getattr(self.actor, "twc", None)
            submodule_detach = getattr(stateful_submodule, "detach", None) if stateful_submodule is not None else None
            if callable(submodule_detach):
                detach_state_fn = submodule_detach

        # Training loop
        actor_losses = []
        critic_losses = []
        entropies = []
        
        for _ in range(epochs):
            if detach_state_fn is not None:
                detach_state_fn()
            # Get current policy outputs
            if hasattr(self.act_space, 'n'):  # Discrete actions
                logits = self.actor(states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            else:  # Continuous actions
                mean, log_std = self.actor(states)
                std = torch.exp(log_std)
                base_dist = torch.distributions.Normal(mean, std)

                # actions are already tanh-squashed; invert them safely
                eps = 1e-6
                a = torch.clamp(actions, -1 + eps, 1 - eps)
                z = 0.5 * torch.log((1 + a) / (1 - a))  # atanh
                
                log_probs = base_dist.log_prob(z) - torch.log(1 - a.pow(2) + eps)
                log_probs = log_probs.sum(dim=-1)
                entropy = base_dist.entropy().sum(dim=-1).mean()  # entropy of base dist
            
            # First get value prediction
            current_values = self.critic(states)
            
            # Compute policy ratio and losses
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()  # Detach advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Add entropy bonus for exploration
            actor_loss = policy_loss - self.entropy_coef * entropy

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Compute value loss
            value_loss = F.mse_loss(current_values, returns)
            scaled_value_loss = self.value_loss_coef * value_loss

            # Update critic
            self.critic_optimizer.zero_grad()
            scaled_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(value_loss.item())
            entropies.append(entropy.item())

        if detach_state_fn is not None:
            detach_state_fn()
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
    
    def set_eval(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
    
    def set_train(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()
    
    def get_save_bundle(self) -> Dict[str, torch.Tensor]:
        """
        Get state dictionaries for saving.
        
        Returns:
            Dictionary containing all state dicts to be saved
        """
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load state dictionaries.
        
        Args:
            state_dict: Dictionary containing state dicts to load
        """
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
