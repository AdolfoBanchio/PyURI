import json
import numpy as np
import os
import torch
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
    # Training loop
    max_episode: int = 300
    max_time_steps: int = 999
    warmup_steps: int = 10_000
    batch_size: int = 128
    num_update_loops: int = 2
    policy_delay: int = 1
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluation
    eval_interval_episodes: int = 10
    eval_episodes: int = 10

    # Exploration noise (OU)
    sigma_start: float = 0.20
    sigma_end: float = 0.05
    sigma_decay_episodes: int = 100

    # BPTT options
    use_bptt: bool = False
    sequence_length: Optional[int] = None
    burn_in_length: Optional[int] = None

    # Saving
    best_model_prefix: str = "td3_actor_best"

    def to_json(self) -> str:
        d = asdict(self)
        # Represent device as string for JSON
        d["device"] = str(self.device)
        return json.dumps(d, indent=4)


def td3_train(env,
              replay_buf,
              ou_noise: OUNoise,
              engine: TD3Engine,
              writer: SummaryWriter,
              timestamp: str,
              config: TD3Config):
    """Unified TD3 training loop supporting standard and BPTT modes.

    - If `config.use_bptt` is True (or `replay_buf` is a `SequenceBuffer`),
      samples sequences and calls `engine.update_step_bptt`.
    - Otherwise, samples transitions and calls `engine.update_step`.
    """

    obs, _ = env.reset()
    best_ret = -np.inf
    total_steps = 0
    os.makedirs("out/models", exist_ok=True)

    for e in tqdm(range(config.max_episode)):
        obs, _ = env.reset()
        ou_noise.update_sigma(e)
        ou_noise.reset()

        ep_reward = 0.0
        steps = 0
        # Reset actor state each episode (works for both stateless/stateful)
        if hasattr(engine.actor, "reset"):
            engine.actor.reset()
        if hasattr(engine, "actor_target") and hasattr(engine.actor_target, "reset"):
            engine.actor_target.reset()

        ep_actions = []
        for _t in range(config.max_time_steps):
            if total_steps < config.warmup_steps:
                action = env.action_space.sample()
            else:
                action = engine.get_action(obs, action_noise=ou_noise.noise)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            steps += 1

            replay_buf.store(obs, action, reward, next_obs, terminated, truncated)
            obs = next_obs
            total_steps += 1

            if done:
                break

            # Updates
            if total_steps > config.warmup_steps and getattr(replay_buf, "size", 0) >= config.batch_size:
                if len(ep_actions) == 0:
                    ep_actions.append(action)

                for _ in range(config.num_update_loops):
                    if config.use_bptt or isinstance(replay_buf, SequenceBuffer):
                        # Guard for sequence hyperparams
                        if not config.sequence_length or not config.burn_in_length:
                            raise ValueError("BPTT enabled but sequence_length/burn_in_length not set")
                        try:
                            seq_batch = replay_buf.sample(config.batch_size, config.sequence_length, config.device)
                            actor_loss, critic_loss = engine.update_step_bptt(seq_batch, config.burn_in_length)
                        except ValueError:
                            # Not enough/long episodes yet; skip this update
                            continue
                    else:
                        batch = replay_buf.sample(config.batch_size, config.device)
                        actor_loss, critic_loss = engine.update_step(batch)

                    # Log losses
                    writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                    writer.add_scalar('Loss/Critic', critic_loss, total_steps)

        # Log episode return and stats
        writer.add_scalar('Training/Episode_Return', ep_reward, total_steps)
        writer.add_scalar('Training/Episode_steps', steps, total_steps)
        if len(ep_actions) > 0:
            writer.add_scalar('Training/AvgAction', float(np.mean(ep_actions)), e)

        # Periodic evaluation
        if (e + 1) % config.eval_interval_episodes == 0:
            eval_ret, eval_avg_action = engine.evaluate_policy(env, episodes=config.eval_episodes)
            writer.add_scalar('Evaluation/Return', eval_ret, total_steps)
            writer.add_scalar('Evaluation/AvgAction', eval_avg_action, total_steps)

            print(f"Evaluation after Episode {e+1}: {eval_ret:.2f}")
            if eval_ret > best_ret:
                best_ret = eval_ret
                prefix = config.best_model_prefix
                torch.save(engine.actor.state_dict(), f"out/models/{prefix}_{timestamp}.pth")
                print(f"New best evaluation reward: {best_ret:.2f}, model saved.")

