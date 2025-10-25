import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from copy import deepcopy


class DDPGEngine():
    """  
    TODO: UPDATE DOCSTRING
    This class functions as a helper to train any actor-critic model with DDPG.
    All objetcts must be already initialized before being passed to the engine.
    Parameters:
        - actor: nn.Module, the policy network.
        - critic: nn.Module, the Q-value network.
        - replay_buffer: ReplayBuffer, the experience replay buffer.
        - actor_optimizer: torch.optim.Optimizer, optimizer for the actor network.
        - critic_optimizer: torch.optim.Optimizer, optimizer for the critic network.
        - device: torch.device, device to run the computations on.
    """
    def __init__(self,
                 gamma: float,
                 tau: float,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 actor: nn.Module,
                 critic: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer,
                 device: torch.device):
        self.gamma = gamma
        self.tau = tau
        self.obs_space = observation_space
        self.act_space = action_space
        self.actor = actor
        self.critic = critic
        self.actor_target = deepcopy(actor)
        self.critic_target = deepcopy(critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.device = device

        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)
        
    def soft_update(self, net: nn.Module, target_net: nn.Module):
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_action(self, state, action_noise = None):
        """  
        Given a state, compute the action to take according to the current policy.
        Optionally add noise for exploration.
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train()
        if action_noise is not None:
            action += action_noise() * 0.5 *( self.act_space.high[0] - self.act_space.low[0])
        
        return np.clip(action, self.act_space.low[0], self.act_space.high[0])
    

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
        done = batch['done'].to(self.device)

        # 1. compute the targets
        with torch.no_grad():
            next_actions = self.actor_target(obs2)
            q_next = self.critic_target(obs2, next_actions)
            q_target = rew + self.gamma * (1 - done) * q_next

        # 2. update Q-function (critic) by one step of gradient descent
        q_current = self.critic(obs, act)
        critic_loss = nn.functional.smooth_l1_loss(q_current, q_target)
        
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 3. update policy (actor) by one step of gradient ascent
        actions_pred = self.actor(obs)
        actor_loss = -self.critic(obs, actions_pred).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 4. update target networks with soft update
        self.soft_update(net=self.actor, target_net=self.actor_target)
        self.soft_update(net=self.critic,  target_net=self.critic_target)

        return actor_loss.item(), critic_loss.item()
    
    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def get_save_bundle(self):
        """  
            returns a dict with all the states dict to be saved
        """
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
