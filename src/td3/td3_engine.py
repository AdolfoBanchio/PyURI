import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from copy import deepcopy

class TD3Engine():
    """  
    TODO: UPDATE DOCSTRING
    
    """
    def __init__(self,
                 gamma: float,
                 tau: float,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 actor: nn.Module,
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
                
    def get_action(self, state, action_noise=None):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(s)
        self.actor.train()

        a = a.squeeze(0)
        if action_noise is not None:
            rng = torch.as_tensor((self.act_space.high - self.act_space.low),
                                  device=self.device, dtype=torch.float32)
            a = a + torch.as_tensor(action_noise(), device=self.device, dtype=torch.float32) * 0.5 * rng

        return torch.clamp(a, self.action_low, self.action_high).cpu().numpy()


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

    def update_step(self, batch):
        """  
        Performs a single update step for both actor and critic networks.
        1. compute the targets
        2. upadte Q-function (critic) by one step of gradient descent
        3. update policy (actor) by one step of gradient ascent
        4. update target networks with soft update

        batch: dict with keys 'obs', 'act', 'rew', 'obs2', 'done' (From ReplayBuffer)
        """
        obs = batch['obs'].to(self.device)
        act = batch['act'].to(self.device)
        rew = batch['rew'].to(self.device)
        obs2 = batch['obs2'].to(self.device)
        term = batch['terminated'].to(self.device).float()
        mask = 1.0 - term

        # 1. compute the targets
        # Ensure recurrent actors do not carry hidden state across random batches
        if hasattr(self.actor_target, 'reset'):
            self.actor_target.reset()
        with torch.no_grad():
            noise = (torch.randn_like(act) * self.target_policy_noise).clamp(-self.target_noise_clip,
                                                                             self.target_noise_clip)
            next_a = self.actor_target(obs2) + noise
            next_a = torch.clamp(next_a, self.action_low, self.action_high)

            q1_t = self.critic_1_target(obs2, next_a)
            q2_t = self.critic_2_target(obs2, next_a)
            min_q = torch.minimum(q1_t, q2_t)

            q_target = rew + self.gamma * mask * min_q

        # 2. update Q-function (critic) by one step of gradient descent
        q1 = self.critic_1(obs, act)
        q2 = self.critic_2(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor (every X steps) ---
        actor_loss = None
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            # Reset policy state before batched actor update as well
            if hasattr(self.actor, 'reset'):
                self.actor.reset()
            
            a = self.actor(obs)

            actor_loss = -self.critic_1(obs, a).mean()
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.soft_update(net=self.actor, target_net=self.actor_target)

            # 4. update target networks with soft update
            self.soft_update(net=self.critic_1,  target_net=self.critic_1_target)
            self.soft_update(net=self.critic_2,  target_net=self.critic_2_target)

        if actor_loss == None:
            actor_loss_r = float('nan')
        else:
            actor_loss_r = actor_loss.item()
            
        return actor_loss_r , critic_loss.item()

    def update_step_bptt(self, batch_of_sequences: dict, burn_in_length: int):
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

        # 2. Burn-in Phase (No Gradients)
        # Reset states for the start of this sequence batch
        self.actor.reset()
        self.actor_target.reset()
        with torch.no_grad():
            for t in range(burn_in_length):
                _ = self.actor(obs_seq[:, t, :])
                _ = self.actor_target(obs2_seq[:, t, :])

        # Detach internal states to stop gradients from flowing into the burn-in period.
        # The actor's state is now detached and ready for the actor-loss unroll.
        if hasattr(self.actor, 'detach'):
            self.actor.detach()
            self.actor_target.detach()

        # 4. Training Phase (With Gradients)
        all_critic_losses = []
        
        # --- OPTIMIZATION: REMOVED STATE CLONING ---
        # The actor's state is already detached and preserved post-burn-in.
        # We no longer need to clone it here.

        for t in range(burn_in_length, L):
            # Get data for this time step
            obs_t = obs_seq[:, t, :]
            act_t = act_seq[:, t, :]
            rew_t = rew_seq[:, t]
            obs2_t = obs2_seq[:, t, :]
            term_t = term_seq[:, t]
            # Ensure shapes are (B, 1)
            rew_t = rew_t.unsqueeze(1)
            mask_t = (1.0 - term_t).unsqueeze(1)

            # --- 4.1. Compute Targets ---
            with torch.no_grad():
                noise = (torch.randn_like(act_t) * self.target_policy_noise).clamp(-self.target_noise_clip,
                                                                                self.target_noise_clip)
                
                # Get next action from target actor, continuing its unroll
                next_a_t = self.actor_target(obs2_t)
                next_a_t = (next_a_t + noise).clamp(self.action_low, self.action_high)
                
                # Critics are stateless
                q1_t = self.critic_1_target(obs2_t, next_a_t)
                q2_t = self.critic_2_target(obs2_t, next_a_t)
                min_q = torch.minimum(q1_t, q2_t)
                
                q_target = rew_t + self.gamma * mask_t * min_q

            # --- 4.2. Update Critic ---
            # Note: self.actor is NOT used in this loop. Its state remains
            # the detached, post-burn-in state, ready for the actor update.
            q1 = self.critic_1(obs_t, act_t)
            q2 = self.critic_2(obs_t, act_t)
            critic_loss_t = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            all_critic_losses.append(critic_loss_t)

        # 5. Backpropagate Critic (BPTT)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = torch.stack(all_critic_losses).mean()
        critic_loss.backward()
        # Clip gradients - ESSENTIAL for recurrent networks
        torch.nn.utils.clip_grad_norm_(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), 1.0)
        self.critic_optimizer.step()

        # --- Actor (every X steps) ---
        actor_loss = None
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            
            # --- OPTIMIZATION: REMOVED STATE RESTORE ---
            # The actor's state is already in the correct post-burn-in,
            # detached state. We can just run the forward pass and
            # BPTT will start from this point, as intended.

            # Temporarily freeze critic params so they are treated as constants
            for p in self.critic_1.parameters():
                p.requires_grad_(False)

            all_actor_losses = []
            for t in range(burn_in_length, L):
                obs_t = obs_seq[:, t, :]
                # This unroll starts from the detached state, building a new graph
                a_t = self.actor(obs_t)
                q_val = self.critic_1(obs_t, a_t) # Uses the UPDATED critic_1
                all_actor_losses.append(-q_val.mean())

            actor_loss = torch.stack(all_actor_losses).mean()

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
            
        return actor_loss_r , critic_loss.item()


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
                a = self.get_action(obs, action_noise=None)
                eval_actions.append(a)
                obs, r, terminated, truncated, _ = env.step(a)
                total += r
                done = terminated or truncated
        self.actor.train()
        return (total / episodes), np.mean(eval_actions)
