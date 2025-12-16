import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import json
import gymnasium as gym
import numpy as np
import torch
import optuna
import ast
import torch.nn.functional as F
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, asdict
from torch.utils.tensorboard import SummaryWriter
from optuna.trial import TrialState
from tqdm import tqdm
from utils import OUNoise, SequenceBuffer
from mlp import TwinCritic
from fiuri import PyUriTwc_V2

@dataclass
class TD3Config:
    # --- Training loop ---
    max_train_steps: int = 300_000
    warmup_steps: int = 10_000
    batch_size: int = 256  # Standard stable batch size for MCC
    num_update_loops: int = 1 # Updates per step (Standard is 1)
    update_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # --- Evaluation ---
    eval_interval_episodes: int = 10
    eval_episodes: int = 10

    # --- BPTT options ---
    sequence_length: int = 8
    burn_in_length: int = 4

    # Hyperparameters
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    
    # OU Noise Parameters
    ou_theta: float = 0.15
    ou_sigma_init: float = 0.5 # Higher initial noise for exploration
    ou_sigma_end: float = 0.1
    ou_sigma_decay_steps: int = 100_000 # Decays over first 1/3 of training
    
    # SG version hyperparameters
    steepness_fire: float = 14.0
    steepness_gj: float = 7.0
    steepness_input: float = 5.0
    input_thresh: float = 0.001

    critic_hidden_layers: int = 256    
    replay_buffer_size: int = 100_000
    
    model_prefix: str = "td3_flat_actor"
    use_SG: bool = True
    
    def to_json(self) -> str:
        d = asdict(self)
        d["critic_hidden_layers"] = str(self.critic_hidden_layers)
        return json.dumps(d, indent=4)

    def load(self, json_data):
        data = json.loads(json_data) if isinstance(json_data, str) else dict(json_data)
        
        # Handle tuple fields
        layers = data.get("critic_hidden_layers")
        if isinstance(layers, str):
            try:
                layers = json.loads(layers)
            except:
                layers = ast.literal_eval(layers)
        if isinstance(layers, (list, tuple)):
            data["critic_hidden_layers"] = tuple(layers)

        for field in self.__dataclass_fields__:
            if field in data:
                setattr(self, field, data[field])
        return self
    
class TD3Engine():
    """  
    TODO: UPDATE DOCSTRING
    
    """
    def __init__(self,
                 gamma: float,
                 tau: float,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 actor: PyUriTwc_V2,
                 critic: TwinCritic,
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
        self.critic = critic
        self.actor_target = deepcopy(actor)
        self.critic_target = deepcopy(critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer  # should include both critics' params
        self.device = device

        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)

        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)

        # cache bounds as tensors
        self.action_low  = torch.as_tensor(self.act_space.low,  device=self.device, dtype=torch.float32)
        self.action_high = torch.as_tensor(self.act_space.high, device=self.device, dtype=torch.float32)

        self.total_updates = 0  # for delayed actor
    
    @torch.no_grad()
    def soft_update(self, target_net, online_net, tau):
        # 1. Soft Update Learnable Parameters (Weights/Biases)
        for target_param, param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        # 2. Hard Copy Non-Learnable Buffers (BatchNorm stats)
        # Buffers (running_mean, running_var) are not "parameters" so they are skipped above.
        for target_buffer, buffer in zip(target_net.buffers(), online_net.buffers()):
            target_buffer.data.copy_(buffer.data)
                
    def get_action(self, state):
        # Shape: (1, ObsDim)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            a = self.actor(s) # Expected (1, ActionDim)

        # Clamp using broadcasted shapes, then squeeze to (ActionDim,)
        return torch.clamp(a, self.action_low, self.action_high).squeeze(0)

    def _detach_state_tuple(self, state_tuple):
        """Helper to detach (E, O) tuple from TWC_V2"""
        return (state_tuple[0].detach(), state_tuple[1].detach())
    
    def update_step_bptt(self, 
                         batch: dict, 
                         burn_in: int,):
        obs = batch['obs'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        terminated = batch['terminated'].to(self.device)
        truncated = batch['truncated'].to(self.device)

        B = obs.shape[0]

        # 1. Burn-in
        with torch.no_grad():
            init_E, init_O = self.actor.get_initial_state(B, self.device)
            if burn_in > 0:
                _, (h_E, h_O) = self.actor.forward_bptt(obs[:, :burn_in], (init_E, init_O))
                _, (h_E_t, h_O_t) = self.actor_target.forward_bptt(next_obs[:, :burn_in], (init_E, init_O))
            else:
                h_E, h_O = init_E, init_O
                h_E_t, h_O_t = init_E, init_O
            
        # Training Sequence Slices
        obs_t = obs[:, burn_in:]
        next_obs_t = next_obs[:, burn_in:]
        act_t = action[:, burn_in:]
        rew_t = reward[:, burn_in:].unsqueeze(-1)
        terminated_t = terminated[:, burn_in:].unsqueeze(-1)
        truncated_t = truncated[:, burn_in:].unsqueeze(-1)
        done_t = terminated_t 

        # 2. Critic Update
        with torch.no_grad():
            # Get next action from target actor
            next_act, _ = self.actor_target.forward_bptt(next_obs_t, (h_E_t, h_O_t))
            
            # Add noise
            noise = (torch.randn_like(next_act) * self.target_policy_noise).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_act = (next_act + noise).clamp(self.action_low, self.action_high)            
            # Target Q
            q1_t, q2_t = self.critic_target(next_obs_t, next_act)
            min_q = torch.min(q1_t, q2_t)            
            target_q = rew_t + (1 - done_t) * self.gamma * min_q

        q1, q2 = self.critic(obs_t, act_t)        
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 3. Actor Update
        actor_loss_val = 0.0
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:

            h_E_detached = h_E.detach()
            h_O_detached = h_O.detach()

            pi_act, _ = self.actor.forward_bptt(obs_t, (h_E_detached, h_O_detached))
            actor_loss = -self.critic.q1_forward(obs_t, pi_act).mean()
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update targets
            self.soft_update(target_net=self.actor_target, 
                             online_net=self.actor, 
                             tau=self.tau)
            self.soft_update(target_net=self.critic_target, 
                             online_net=self.critic, 
                             tau=self.tau)
            
            actor_loss_val = actor_loss.item()
            
        return actor_loss_val, critic_loss.item(), 
    
    @torch.no_grad()
    def evaluate_policy(self, env, episodes=10):
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

def phi(obs):
    pos = obs[0]
    pos_min, pos_max = -1.2, 0.6
    x = (pos - pos_min) / (pos_max - pos_min)
    return 5.0 * float(np.clip(x, 0.0, 1.0))

def td3_train(
    env: gym.Env,
    replay_buf: SequenceBuffer,
    engine: TD3Engine,
    writer: SummaryWriter,
    timestamp: str,
    config: TD3Config,
    OUNoise: OUNoise,
    trial: optuna.Trial = None,
):
    """
    Main training loop for TD3, adapted to run for a maximum number of time steps.
    """
    total_steps = 0
    best_ret = -np.inf
    best_model_path = None
    e = 0  # Episode counter

    # loop variables
    env_seed = config.seed
    max_train_steps = config.max_train_steps
    warmup_steps =  config.warmup_steps
    num_update_loops = config.num_update_loops
    update_every_steps = config.update_every
    batch_size = config.batch_size
    sequence_length = config.sequence_length
    device = config.device
    burn_in_length = config.burn_in_length
    eval_interval_episodes =  config.eval_interval_episodes
    eval_episodes =  config.eval_episodes
    model_prefix = config.model_prefix
    action_low  = torch.as_tensor(engine.act_space.low,  device=engine.device, dtype=torch.float32)
    action_high = torch.as_tensor(engine.act_space.high, device=engine.device, dtype=torch.float32)
    
    print(f"working on device: {device}")
    # Use tqdm to track total steps
    pbar = tqdm(total=config.max_train_steps, initial=total_steps, desc="Training TD3")

    eval_idx = 0
    all_eval_scores = []
    pruning_window = deque(maxlen=3)

    while total_steps < max_train_steps:
        # New episode
        obs, _ = env.reset(seed=env_seed)
        ep_reward = 0.0
        ep_actions = []
        steps = 0

        OUNoise.reset()

        engine.actor.reset()
        engine.actor_target.reset()
        
        done = False

        while not done:
            if total_steps >= max_train_steps:
                break
            # Action Selection
            a_det = engine.get_action(obs)

            if total_steps < warmup_steps: 
                action = env.action_space.sample()
            else: 
                """  
                this get action, makes a forward pass where the actor maintain his own state
                through all the active training episode. In the update step network internal state
                is managed differntly to avoid intervinig in the state of the network during
                the active episode. 
                """                
                OUNoise.update(total_steps=total_steps)
                noise = torch.as_tensor(OUNoise.noise(), 
                                            device=a_det.device,
                                            dtype=a_det.dtype)

                action = torch.clamp(a_det + noise, action_low, action_high).detach().cpu().numpy()

            # Environment step 
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_actions.append(action[0])
            
            # Store transition and update step counters 
            shaped = reward + engine.gamma * phi(obs2) - phi(obs)
            replay_buf.store(obs, action, shaped, obs2, terminated, truncated)
            obs = obs2
            total_steps += 1
            steps += 1
            
            # Update every X episodes 
            if total_steps > warmup_steps and (total_steps % update_every_steps == 0):
                for _ in range(num_update_loops):
                    seq_batch = replay_buf.sample(batch_size, sequence_length, device)
                    actor_loss, critic_loss = engine.update_step_bptt(seq_batch, burn_in_length)

                # Log losses every 100 steps to avoid exessive IO
                if total_steps % 100 == 0:
                    writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                    writer.add_scalar('Loss/Critic', critic_loss, total_steps)
                    if OUNoise:
                        writer.add_scalar('Training/OUNoise_sigma', OUNoise.sigma, total_steps)
        
        pbar.update(steps) # Update the tqdm progress bar
        
        if done:
            # Current ep ended, log middle trainig results
            writer.add_scalar('Training/Episode_Return', ep_reward, total_steps)
            writer.add_scalar('Training/Episode_steps', steps, total_steps)
            if len(ep_actions) > 0:
                writer.add_scalar('Training/AvgAction', float(np.mean(ep_actions)), total_steps)
                writer.add_scalar('Training/StdAction', float(np.std(ep_actions)), total_steps)
            
            e += 1

            # Evaluation & Optuna Pruning
            if e % eval_interval_episodes == 0:
                eval_ret, eval_avg_action = engine.evaluate_policy(env, episodes=eval_episodes)
                writer.add_scalar('Evaluation/Return', eval_ret, total_steps)
                writer.add_scalar('Evaluation/AvgAction', eval_avg_action, total_steps)

                tqdm.write(f"\nEpisode {e}: TotalSteps: {total_steps}, EvalReturn: {eval_ret:.2f}")
                all_eval_scores.append(eval_ret)
                pruning_window.append(eval_ret)
                rolling_avg = np.mean(pruning_window)
                if eval_ret > best_ret:
                    best_ret = eval_ret
                    prefix = model_prefix
                    model_path = os.path.join(writer.log_dir, f"{prefix}_best_{timestamp}.pth")
                    torch.save(engine.actor.state_dict(), model_path)
                    best_model_path = model_path
                    tqdm.write(f"New best evaluation reward: {best_ret:.2f}. Model saved to {model_path}")
                
                # --- OPTUNA PRUNING LOGIC ---
                if trial is not None:
                    trial.report(rolling_avg, step=eval_idx)
                    if trial.should_prune():
                        tqdm.write(f"Trial {trial.number} pruned at eval {eval_idx} (episode {e}) with best return {best_ret:.2f}.")
                        pbar.close()
                        raise optuna.TrialPruned()
                eval_idx += 1


    pbar.close()
    
    # --- Final Save ---
    prefix = f"{model_prefix}_final"
    model_path = os.path.join(writer.log_dir, f"{prefix}_{timestamp}.pth")            
    torch.save(engine.actor.state_dict(), model_path)
    print(f"Final Model saved to {model_path}")           
    return all_eval_scores, best_model_path
