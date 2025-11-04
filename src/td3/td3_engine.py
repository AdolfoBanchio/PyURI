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
            
            actor_loss = -self.critic_1(obs, self.actor(obs)).mean()
            
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
