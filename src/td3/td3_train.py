import json
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
    max_episode: int = 300
    max_train_steps: int = 300_000
    max_time_steps_per_ep: int = 999
    warmup_steps: int = 10_000
    batch_size: int = 128
    num_update_loops: int = 2 
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Evaluation ---
    eval_interval_episodes: int = 10
    eval_episodes: int = 10

    # --- BPTT options ---
    use_bptt: bool = False
    sequence_length: Optional[int] = None
    burn_in_length: Optional[int] = None

    # --- Saving ---
    best_model_prefix: str = "td3_actor_best"
    
    # --- Optuna-Tunable Hyperparameters (with defaults) ---
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
    critic_hidden_layers: list[int] = (128, 128)
    
    # Buffer
    replay_buffer_size: int = 100_000
    replay_buffer_keep: int = 20_000 # For sequence buffer

    def to_json(self) -> str:
        d = asdict(self)
        d["device"] = str(self.device)
        d["critic_hidden_layers"] = str(self.critic_hidden_layers) # Convert tuple/list
        return json.dumps(d, indent=4)


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
    Main training loop for TD3.
    """
    total_steps = 0
    best_ret = -np.inf
    e = 0
    for s in tqdm(range(config.max_episode)):
        obs, _ = env.reset()
        ou_noise.reset()
        ou_noise.update_sigma(e)

        ep_reward = 0.0
        ep_actions = []
        steps = 0
        if hasattr(engine.actor, 'reset'):
            engine.actor.reset()
            engine.actor_target.reset()

        for t in range(config.max_time_steps_per_ep):
            if total_steps < config.warmup_steps:
                action = env.action_space.sample()
            else:
                action = engine.get_action(obs, ou_noise.noise)
             
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_actions.append(action[0])
            
            replay_buf.store(obs, action, reward, obs2, terminated, truncated)
            old_obs = obs
            obs = obs2
            total_steps += 1
            steps += 1
            
            # --- Update Phase ---
            if total_steps > config.warmup_steps:
                for _ in range(config.num_update_loops):
                    try:
                        if config.use_bptt:
                            seq_batch = replay_buf.sample(
                                config.batch_size, config.sequence_length, config.device
                            )
                            actor_loss, critic_loss = engine.update_step_bptt(
                                seq_batch, config.burn_in_length
                            )
                        else:
                            batch = replay_buf.sample(config.batch_size, config.device)
                            actor_loss, critic_loss = engine.update_step(batch)

                        # Log losses
                        writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                        writer.add_scalar('Loss/Critic', critic_loss, total_steps)
                    
                    except ValueError as e:
                        # Handle SequenceBuffer not being ready
                        if config.use_bptt:
                            # print(f"Skipping update: {e}")
                            pass
                        else:
                            raise e

            if done:
                break
        
        # --- End of Episode ---
        writer.add_scalar('Training/Episode_Return', ep_reward, total_steps)
        writer.add_scalar('Training/Episode_steps', steps, total_steps)
        if len(ep_actions) > 0:
            writer.add_scalar('Training/AvgAction', float(np.mean(ep_actions)), e)
            writer.add_scalar('Training/StdAction', float(np.std(ep_actions)), e)

        # --- Periodic Evaluation & Optuna Pruning ---
        if (e + 1) % config.eval_interval_episodes == 0:
            eval_ret, eval_avg_action = engine.evaluate_policy(env, episodes=config.eval_episodes)
            writer.add_scalar('Evaluation/Return', eval_ret, total_steps)
            writer.add_scalar('Evaluation/AvgAction', eval_avg_action, total_steps)
        
            print(f"\nEpisode {e+1}: TotalSteps: {total_steps}, EvalReturn: {eval_ret:.2f}")

            if eval_ret > best_ret:
                best_ret = eval_ret
                prefix = config.best_model_prefix
                model_path = os.path.join(writer.log_dir, f"{prefix}_best_{timestamp}.pth")
                torch.save(engine.actor.state_dict(), model_path)
                print(f"New best evaluation reward: {best_ret:.2f}. Model saved to {model_path}")
            
            # --- OPTUNA PRUNING LOGIC ---
            if trial is not None:
                trial.report(eval_ret, e + 1)
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at episode {e+1} with return {eval_ret}.")
                    raise optuna.TrialPruned()
    
    prefix = "td3_actor_final_bptt" if config.use_bptt else "td3_actor_final"
    model_path = os.path.join(writer.log_dir, f"{prefix}_{timestamp}.pth")            
    torch.save(engine.actor.state_dict(), model_path)
    print(f"Final Model saved to {model_path}")           
    return best_ret # Return the best evaluation reward for this trial

def td3_train_by_steps(
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

    # Use tqdm to track total steps
    pbar = tqdm(total=config.max_train_steps, initial=total_steps, desc="Training TD3")

    # Main loop runs until total_steps reaches max_train_steps
    while total_steps < config.max_train_steps:
        # --- Episode Start ---
        obs, _ = env.reset()
        ou_noise.reset()
        ou_noise.update_sigma(e)

        ep_reward = 0.0
        ep_actions = []
        steps = 0
        if hasattr(engine.actor, 'reset'):
            engine.actor.reset()
            engine.actor_target.reset()

        # Inner loop runs until episode terminates or max steps per episode is reached
        for t in range(config.max_time_steps_per_ep):
            if total_steps >= config.max_train_steps:
                # Break out of the inner loop if max steps is reached
                break
                
            # --- Action Selection ---
            if total_steps < config.warmup_steps:
                action = env.action_space.sample()
            else:
                action = engine.get_action(obs, ou_noise.noise)
             
            # --- Environment Step ---
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_actions.append(action[0])
            
            # --- Store Transition and Update Step Counters ---
            replay_buf.store(obs, action, reward, obs2, terminated, truncated)
            old_obs = obs # Not used in this snippet, but kept for consistency
            obs = obs2
            total_steps += 1
            steps += 1
            pbar.update(1) # Update the tqdm progress bar
            
            # --- Update Phase ---
            if total_steps > config.warmup_steps:
                for _ in range(config.num_update_loops):
                    try:
                        # Logic for BPTT or standard update remains the same
                        if config.use_bptt:
                            seq_batch = replay_buf.sample(
                                config.batch_size, config.sequence_length, config.device
                            )
                            actor_loss, critic_loss = engine.update_step_bptt(
                                seq_batch, config.burn_in_length
                            )
                        else:
                            batch = replay_buf.sample(config.batch_size, config.device)
                            actor_loss, critic_loss = engine.update_step(batch)

                        # Log losses
                        writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                        writer.add_scalar('Loss/Critic', critic_loss, total_steps)
                    
                    except ValueError as e:
                        # Handle SequenceBuffer not being ready
                        if config.use_bptt:
                            pass
                        else:
                            raise e

            if done:
                break
        
        # --- End of Episode (if it completed) ---
        if total_steps <= config.max_train_steps:
            writer.add_scalar('Training/Episode_Return', ep_reward, total_steps)
            writer.add_scalar('Training/Episode_steps', steps, total_steps)
            if len(ep_actions) > 0:
                writer.add_scalar('Training/AvgAction', float(np.mean(ep_actions)), e)
                writer.add_scalar('Training/StdAction', float(np.std(ep_actions)), e)

            # --- Periodic Evaluation & Optuna Pruning ---
            if (e + 1) % config.eval_interval_episodes == 0:
                eval_ret, eval_avg_action = engine.evaluate_policy(env, episodes=config.eval_episodes)
                writer.add_scalar('Evaluation/Return', eval_ret, total_steps)
                writer.add_scalar('Evaluation/AvgAction', eval_avg_action, total_steps)
            
                print(f"\nEpisode {e+1}: TotalSteps: {total_steps}, EvalReturn: {eval_ret:.2f}")

                if eval_ret > best_ret:
                    best_ret = eval_ret
                    prefix = config.best_model_prefix
                    model_path = os.path.join(writer.log_dir, f"{prefix}_best_{timestamp}.pth")
                    torch.save(engine.actor.state_dict(), model_path)
                    print(f"New best evaluation reward: {best_ret:.2f}. Model saved to {model_path}")
                
                # --- OPTUNA PRUNING LOGIC ---
                if trial is not None:
                    # Report using the episode count for Optuna, but use total_steps for logging
                    trial.report(eval_ret, e + 1) 
                    if trial.should_prune():
                        print(f"Trial {trial.number} pruned at episode {e+1} with return {eval_ret}.")
                        # Close the progress bar before raising exception
                        pbar.close() 
                        raise optuna.TrialPruned()
            
            # Increment episode counter
            e += 1

    pbar.close()
    
    # --- Final Save ---
    prefix = "td3_actor_final_bptt" if config.use_bptt else "td3_actor_final"
    model_path = os.path.join(writer.log_dir, f"{prefix}_{timestamp}.pth")            
    torch.save(engine.actor.state_dict(), model_path)
    print(f"Final Model saved to {model_path}")           
    return best_ret