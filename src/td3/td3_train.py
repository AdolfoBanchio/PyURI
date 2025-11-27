import json
import ast
import numpy as np
import os
import torch
import optuna
import gymnasium as gym
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .td3_engine import TD3Engine
from utils.replay_buffer import ReplayBuffer
from utils.sequence_buffer import SequenceBuffer
from utils.ou_noise import OUNoise


@dataclass
class TD3Config:
    # --- Training loop ---
    max_train_steps: int = 300_000
    max_episode_steps: int = 999
    warmup_steps: int = 10_000
    batch_size: int = 128
    num_update_loops: int = 2 
    update_every: int = 1
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    # --- Evaluation ---
    eval_interval_episodes: int = 10
    eval_episodes: int = 10

    # --- BPTT options ---
    use_bptt: bool = False
    sequence_length: Optional[int] = None
    burn_in_length: Optional[int] = None

    # Hyperparameters
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    
    # Exploration noise (OU)
    sigma_start: float = 0.20
    sigma_end: float = 0.05
    sigma_decay_episodes: int = 125

    # Model-specific (TWC and Critic)
    twc_internal_steps: int = 3
    twc_trhesholds: list[float] = (-0.5, 0.0, 0.0)
    twc_decays: list[float] = (2.2, 0.1, 0.1)
    rnd_init: bool = False
    use_v2: bool = False

    # SG version hyperparameters
    steepness_fire: float = 14
    steepness_gj: float = 7
    steepness_input: float = 5
    input_thresh: float = 0.001
    leaky_slope: float = 0.02

    critic_hidden_layers: list[int] = (400, 300)
    
    # Buffer
    replay_buffer_size: int = 100_000
    replay_buffer_keep: int = 20_000 # For sequence buffer

    # --- Saving ---
    model_prefix: str = "td3_actor"
    
    def to_json(self) -> str:
        d = asdict(self)
        d["device"] = str(self.device)
        d["critic_hidden_layers"] = str(self.critic_hidden_layers) # Convert tuple/list
        return json.dumps(d, indent=4)

    def load(self, json_data):
        data = json.loads(json_data) if isinstance(json_data, str) else dict(json_data)

        # Normalize device
        if "device" in data:
            data["device"] = torch.device(data["device"])

        # Parse critic layers that may have been serialized as a string
        layers = data.get("critic_hidden_layers")
        if isinstance(layers, str):
            try:
                layers = json.loads(layers)
            except json.JSONDecodeError:
                layers = ast.literal_eval(layers)
        if isinstance(layers, (list, tuple)):
            data["critic_hidden_layers"] = tuple(layers)

        for field in self.__dataclass_fields__:
            if field in data:
                setattr(self, field, data[field])
        return self

def td3_train(
    env: gym.Env,
    replay_buf: ReplayBuffer,
    ou_noise: OUNoise,
    engine: TD3Engine,
    writer: SummaryWriter,
    timestamp: str,
    config: TD3Config,
    trial: optuna.Trial = None
):
    """
    Main training loop for TD3, adapted to run for a maximum number of time steps.
    """
    total_steps = 0
    best_ret = -np.inf
    e = 0  # Episode counter

    # loop variables
    env_seed = config.seed
    max_train_steps = config.max_train_steps
    warmup_steps =  config.warmup_steps
    num_update_loops = config.num_update_loops
    update_every_steps = config.update_every
    use_bptt = config.use_bptt
    batch_size = config.batch_size
    sequence_length = config.sequence_length
    device = config.device
    burn_in_length = config.burn_in_length
    eval_interval_episodes =  config.eval_interval_episodes
    eval_episodes =  config.eval_episodes
    model_prefix = config.model_prefix

    # Use tqdm to track total steps
    pbar = tqdm(total=config.max_train_steps, initial=total_steps, desc="Training TD3")

    while total_steps < max_train_steps:
        # New episode
        obs, _ = env.reset(seed=env_seed)
        # env_seed += 1 for now avoid modifying seed during trainig for reproducibility
        ou_noise.reset()
        ou_noise.update_sigma(e)

        ep_reward = 0.0
        ep_actions = []
        steps = 0

        if hasattr(engine.actor, 'reset'):
            engine.actor.reset()
            engine.actor_target.reset()
        
        done = False

        while not done:
            if total_steps >= max_train_steps:
                break
            # Action Selection
            if total_steps < warmup_steps: 
                action = env.action_space.sample()
            else: 
                action = engine.get_action(obs, ou_noise.noise)
             
            # Environment step 
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_actions.append(action[0])
            
            # Store transition and update step counters 
            replay_buf.store(obs, action, reward, obs2, terminated, truncated)
            obs = obs2
            total_steps += 1
            steps += 1
            
            # Update every X episodes 
            if total_steps > warmup_steps and (total_steps % update_every_steps == 0):
                for _ in range(num_update_loops):
                    # Logic for BPTT or standard update remains the same
                    if use_bptt:
                        seq_batch = replay_buf.sample(batch_size, sequence_length, device)
                        actor_loss, critic_loss = engine.update_step_bptt(seq_batch, burn_in_length)
                    else:
                        batch = replay_buf.sample(batch_size, device)
                        actor_loss, critic_loss = engine.update_step(batch)
                    
                    # Log losses every 100 steps to avoid exessive IO
                    if total_steps % 100 == 0:
                        writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                        writer.add_scalar('Loss/Critic', critic_loss, total_steps)
        
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
                
                if eval_ret > best_ret:
                    best_ret = eval_ret
                    prefix = model_prefix
                    model_path = os.path.join(writer.log_dir, f"{prefix}_best_{timestamp}.pth")
                    torch.save(engine.actor.state_dict(), model_path)
                    tqdm.write(f"New best evaluation reward: {best_ret:.2f}. Model saved to {model_path}")
                
                # --- OPTUNA PRUNING LOGIC ---
                if trial is not None:
                    # Report using the episode count for Optuna, but use total_steps for logging
                    trial.report(eval_ret, total_steps) 
                    if trial.should_prune():
                        tqdm.write(f"Trial {trial.number} pruned at episode {e} with return {eval_ret}.")
                        pbar.close() 
                        raise optuna.TrialPruned()

    pbar.close()
    
    # --- Final Save ---
    prefix = model_prefix + "_final_bptt" if use_bptt else model_prefix + "_final"
    model_path = os.path.join(writer.log_dir, f"{prefix}_{timestamp}.pth")            
    torch.save(engine.actor.state_dict(), model_path)
    print(f"Final Model saved to {model_path}")           
    return best_ret
