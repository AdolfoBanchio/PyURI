import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from copy import deepcopy
from twc.twc_builder import TWC
from typing import Optional

class TD3Engine():
    """  
    TODO: UPDATE DOCSTRING
    
    """
    def __init__(self,
                 gamma: float,
                 tau: float,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 actor: TWC,
                 critic_1: nn.Module,
                 critic_2: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer,
                 policy_delay: int = 2,
                 target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5,
                 device: torch.device = torch.device("cpu")):
        
        self.gamma = gamma
        self.tau = tau
        self.obs_space = observation_space
        self.act_space = action_space
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.actor_target = deepcopy(actor)
        self.critic_1_target = deepcopy(critic_1)
        self.critic_2_target = deepcopy(critic_2)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer  # should include both critics' params
        self.device = device

        self.actor.to(device)
        self.critic_1.to(device)
        self.critic_2.to(device)
        self.actor_target.to(device)
        self.critic_1_target.to(device)
        self.critic_2_target.to(device)

        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)

        # cache bounds as tensors
        self.action_low  = torch.as_tensor(self.act_space.low,  device=self.device, dtype=torch.float32)
        self.action_high = torch.as_tensor(self.act_space.high, device=self.device, dtype=torch.float32)

        self.total_updates = 0  # for delayed actor
    
    @torch.no_grad()
    def soft_update(self, net: nn.Module, target_net: nn.Module):
            for p, tp in zip(net.parameters(), target_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
                
    def get_action(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            a = self.actor(s)

        return torch.clamp(a, self.action_low, self.action_high).squeeze(0)


    def set_eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.actor_target.eval()
        self.critic_1_target.eval()
        self.critic_2_target.eval()
    
    def set_train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.actor_target.train()
        self.critic_1_target.train()
        self.critic_2_target.train()

    def _detach_state_dict(self, state_dict: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        """
        Helper method: Detaches all tensors in a state dictionary from the graph.
        """
        return {
            name: (state_pair[0].detach(), state_pair[1].detach())
            for name, state_pair in state_dict.items()
        }
    
    def update_step_bptt(self, 
                         batch_of_sequences: dict, 
                         burn_in_length: int,
                         is_weights: Optional[torch.Tensor] = None # To use with PER
                         ):
        """
        Performs a single update step for both actor and critic networks
        using Truncated Backpropagation Through Time (BPTT).

        This version is optimized to remove all redundant state cloning,
        relying on the BPTT-standard 'detach' to truncate the graph.
        """
        
        # 1. Get sequence data and dimensions
        obs_seq = batch_of_sequences['obs'].to(self.device)
        act_seq = batch_of_sequences['action'].to(self.device)
        rew_seq = batch_of_sequences['reward'].to(self.device)
        obs2_seq = batch_of_sequences['next_obs'].to(self.device)
        term_seq = batch_of_sequences['done'].to(self.device) # Use 'done' from buffer
        
        B, L, _ = obs_seq.shape
        train_length = L - burn_in_length
        assert train_length > 0, "Sequence length must be > burn_in_length"

        # Importance Sampling weights (B, 1)
        if is_weights is None:
            is_weights = torch.ones((B, 1), device=self.device, dtype=torch.float32)
        else:
            is_weights = is_weights.to(self.device)
        # 2. Burn-in Phase (No Gradients)
        # Reset states for the start of this sequence batch
        state_a = self.actor.get_initial_state(B, self.device)    
        state_at = self.actor_target.get_initial_state(B, self.device)

        with torch.no_grad():
            for t in range(burn_in_length):
                _, state_a = self.actor.forward_bptt(obs_seq[:, t, :], state_a)
                _, state_at = self.actor_target.forward_bptt(obs2_seq[:, t, :], state_at)

        # Detach internal states to stop gradients from flowing into the burn-in period.
        # The actor's state is now detached and ready for the actor-loss unroll.
        state_a = self._detach_state_dict(state_a)
        state_at = self._detach_state_dict(state_at)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = 0.0
        td_error_accum = torch.zeros(B, device=self.device)

        for t in range(burn_in_length, L):
            obs_t = obs_seq[:, t, :] 
            act_t = act_seq[:, t, :]
            rew_t = rew_seq[:, t].unsqueeze(1)
            obs2_t = obs2_seq[:, t, :]
            mask_t = (1.0 - term_seq[:, t]).unsqueeze(1)

            with torch.no_grad():
                noise = (
                    torch.randn_like(act_t) * self.target_policy_noise
                    ).clamp(-self.target_noise_clip, self.target_noise_clip)
                next_a_t, state_at = self.actor_target.forward_bptt(obs2_t, state_at)
                next_a_t = (next_a_t + noise).clamp(self.action_low, self.action_high)

                q1_t = self.critic_1_target(obs2_t, next_a_t)
                q2_t = self.critic_2_target(obs2_t, next_a_t)
                min_q = torch.minimum(q1_t, q2_t)

                q_target = rew_t + self.gamma * mask_t * min_q

            q1 = self.critic_1(obs_t, act_t)
            q2 = self.critic_2(obs_t, act_t)

            # TD-error for PER
            td_t = (q1.detach() - q_target.detach()).abs().squeeze(-1)  # (B,)
            td_error_accum += td_t

            # Critic loss por muestras (sin reduction)
            mse1 = (q1 - q_target).pow(2)  # (B, 1)
            mse2 = (q2 - q_target).pow(2)  # (B, 1)
            per_sample_loss = mse1 + mse2  # (B, 1)

            # Aplicar importance sampling weights (broadcast (B,1))
            per_sample_loss = per_sample_loss * is_weights

            # Acumular loss (promedio sobre batch)
            critic_loss = critic_loss + per_sample_loss.mean()

        # Average the loss and backpropagate
        critic_loss = critic_loss / train_length
        critic_loss.backward()
        # Clip gradients - ESSENTIAL for recurrent networks
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), 1.0
            )
        self.critic_optimizer.step()

        td_errors_seq = td_error_accum / train_length   # (B,)

        # --- Actor (every X steps) ---
        actor_loss = None
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            # Temporarily freeze critic params so they are treated as constants
            for p in self.critic_1.parameters():
                p.requires_grad_(False)

            actor_loss = 0.0
            for t in range(burn_in_length, L):
                obs_t = obs_seq[:, t, :]
                a_t, state_a = self.actor.forward_bptt(obs_t, state_a)
                q_val = self.critic_1(obs_t, a_t)
                
                actor_loss = actor_loss - q_val.mean()

            actor_loss = actor_loss / (L - burn_in_length)

            # Restore critic params to require grad for future critic updates
            for p in self.critic_1.parameters():
                p.requires_grad_(True)

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft-update targets
            self.soft_update(net=self.actor, target_net=self.actor_target)
            self.soft_update(net=self.critic_1,  target_net=self.critic_1_target)
            self.soft_update(net=self.critic_2,  target_net=self.critic_2_target)

        if actor_loss is None:
            actor_loss_r = float('nan')
        else:
            actor_loss_r = actor_loss.item()
            
        return actor_loss_r , critic_loss.item(), td_errors_seq


    @torch.no_grad()
    def evaluate_policy(self,env, episodes=10):
        self.actor.eval()
        total = 0.0
        eval_actions = []
        for _ in range(episodes):
            obs, _ = env.reset()
            self.actor.reset()
            done = False
            while not done:
                a = self.get_action(obs)
                a_np = a.detach().cpu().numpy()
                eval_actions.append(a_np)
                obs, r, terminated, truncated, _ = env.step(a_np)
                total += r
                done = terminated or truncated
        self.actor.train()
        return (total / episodes), np.mean(eval_actions)
