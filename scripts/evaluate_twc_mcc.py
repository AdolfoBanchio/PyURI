import argparse
import csv
import os
import sys
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from gymnasium.wrappers import RecordVideo

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlp import Actor
from twc import mcc_obs_encoder, twc_out_2_mcc_action, build_twc
from td3 import TD3Config

ENV = "MountainCarContinuous-v0"
SEED = 42
DEFAULT_EPISODES = 100
DEFAULT_ACTOR_HIDDEN = [400, 300]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPORTS_DIR = "out/reports"
VIDEOS_DIR = "out/videos"

REPORTS_FILEPATH = "out/reports/twc_td3_reports.csv"

def parse_config(cfg: TD3Config, model_path: Path):
    v2_params = {}
    if cfg.use_v2:
        v2_params = {
            'steepness_fire': cfg.steepness_fire,
            'steepness_gj': cfg.steepness_gj,
            'steepness_input': cfg.steepness_input,
            'input_thresh': cfg.input_thresh,
            'leaky_slope': cfg.leaky_slope,
            }
    actor = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_action,
        internal_steps=cfg.twc_internal_steps,
        rnd_init=cfg.rnd_init,
        use_V2=cfg.use_v2,
        log_stats=True,
        **v2_params
    )

    state_dict = torch.load(model_path, map_location=DEVICE)
    actor.load_state_dict(state_dict=state_dict)
    actor.to(DEVICE)
    actor.eval()
    return actor

def evaluate_model(env: gym.Env, actor: torch.nn.Module, n_eps: int = DEFAULT_EPISODES):
    """Roll out the policy for n_eps episodes and collect summary statistics."""
    rewards, steps = [], []
    best_reward = -np.inf
    best_seed = None

    @torch.no_grad()
    def policy(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action = actor(obs_t).squeeze(0).cpu().numpy()
        return action

    for ep in range(n_eps):
        seed = SEED + ep
        obs, _ = env.reset(seed=seed)
        if hasattr(actor, "reset"):
            actor.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

        rewards.append(ep_reward)
        steps.append(ep_steps)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_seed = seed

    rewards_np = np.array(rewards, dtype=np.float32)
    steps_np = np.array(steps, dtype=np.int32)
    metrics = {
        "episodes": n_eps,
        "mean_reward": float(rewards_np.mean()),
        "std_reward": float(rewards_np.std(ddof=1) if n_eps > 1 else 0.0),
        "mean_steps": float(steps_np.mean()),
        "std_steps": float(steps_np.std(ddof=1) if n_eps > 1 else 0.0),
        "min_steps": int(steps_np.min()),
        "max_steps": int(steps_np.max()),
        "best_reward": float(best_reward),
        "best_seed": int(best_seed) if best_seed is not None else None,
        "per_episode_rewards": rewards,
        "per_episode_steps": steps,
    }
    return metrics


def record_episode(
    model_name: str, actor: torch.nn.Module, seed: int, video_dir: Path
) -> Optional[str]:
    """Record a single episode with the provided seed and return the video file path."""
    if seed is None:
        return None

    if hasattr(actor, "log"):
        actor.log = True

    os.makedirs(video_dir, exist_ok=True)
    base_env = gym.make(ENV, render_mode="rgb_array")
    record_env = RecordVideo(
        base_env,
        video_folder=str(video_dir),
        episode_trigger=lambda idx: True,
        name_prefix=f"{model_name}_best",
    )

    @torch.no_grad()
    def policy(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action = actor(obs_t).squeeze(0).cpu().numpy()
        return action

    obs, _ = record_env.reset(seed=seed)
    if hasattr(actor, "reset"):
        actor.reset()
    done = False
    while not done:
        action = policy(obs)
        obs, _, terminated, truncated, _ = record_env.step(action)
        done = terminated or truncated

    video_name = getattr(record_env, "_video_name", None)
    video_path = str(video_dir / f"{video_name}.mp4") if video_name else None
    record_env.close()
    return video_path


def write_report(
    metrics: dict,
    model_path: Path,
    reports_filepath: Path,
    config: TD3Config,
    video_path: Optional[str],
):
    model_name = model_path.stem
    reports_filepath.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp",
        "version",
        "model_name",
        "model_path",
        "env",
        "episodes",
        "mean_reward",
        "std_reward",
        "mean_steps",
        "std_steps",
        "min_steps",
        "max_steps",
        "best_reward",
        "best_seed",
        "video_path",
    ]

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "V2" if config.use_v2 else "V1",
        "model_name": model_name,
        "model_path": str(model_path),
        "env": ENV,
        "episodes": metrics["episodes"],
        "mean_reward": metrics["mean_reward"],
        "std_reward": metrics["std_reward"],
        "mean_steps": metrics["mean_steps"],
        "std_steps": metrics["std_steps"],
        "min_steps": metrics["min_steps"],
        "max_steps": metrics["max_steps"],
        "best_reward": metrics["best_reward"],
        "best_seed": metrics["best_seed"],
        "video_path": video_path or "",
    }

    existing_models = set()
    if reports_filepath.exists():
        with reports_filepath.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for existing_row in reader:
                existing_models.add(existing_row.get("model_path") or existing_row.get("model_name"))

    if str(model_path) in existing_models or model_name in existing_models:
        return reports_filepath

    write_header = not reports_filepath.exists() or reports_filepath.stat().st_size == 0
    with reports_filepath.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return reports_filepath

def save_twc_plots(twc, out_dir="out/videos"):
    # Extract monitor logs into (T, N) tensors per layer
    monitor = twc.monitor
    layers = ['in', 'hid', 'out']
    series = {}
    for L in layers:
        in_states = torch.stack([step['in_state'][0] for step in monitor[L]], dim=0)  # (T, N)
        out_states = torch.stack([step['out_state'][0] for step in monitor[L]], dim=0)  # (T, N)
        series[L] = (in_states, out_states)

    # Plot per-layer time series for in/out state of each neuron
    for L in layers:
        in_states, out_states = series[L]
        N = in_states.shape[1]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        for i in range(N):
            ax1.plot(in_states[:, i].cpu().numpy(), label=f'E_{i}')
            ax2.plot(out_states[:, i].cpu().numpy(), label=f'O_{i}')
        ax1.set_title(f'{L.upper()} layer: Internal state (E) per neuron')
        ax2.set_title(f'{L.upper()} layer: Output state (O) per neuron')
        ax2.set_xlabel('Time step')
        ax1.set_ylabel('E')
        ax2.set_ylabel('O')
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        # Keep legends compact
        ax1.legend(ncol=max(1, N // 4), fontsize='small')
        ax2.legend(ncol=max(1, N // 4), fontsize='small')
        plt.tight_layout()
        save_path = os.path.join(out_dir, f'twc_{L}_states.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f'Saved {save_path}')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a MountainCarContinuous policy and export metrics."
    )
    parser.add_argument("model_path", type=str, help="Path to the actor state_dict to evaluate.")
    parser.add_argument(
        "--config",
        type=str,
        help=f"Path to the trial configuration (.json file)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    reports_dir = Path(REPORTS_FILEPATH)
    videos_dir = Path(VIDEOS_DIR)
    model_path = Path(args.model_path)
    config_path = Path(args.config)

    cfg = TD3Config()
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data =  json.load(f)
        cfg = cfg.load(config_data)
    
    print(cfg.to_json())

    actor = parse_config(cfg=cfg, model_path=model_path)
    env = gym.make(ENV)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    metrics = evaluate_model(env, actor, n_eps=DEFAULT_EPISODES)
    env.close()

    video_path = None
    video_path = record_episode(model_path.stem, actor, metrics["best_seed"], videos_dir)
    print(video_path)
    
    #if "twc" in model_path.name.lower():
    #    save_twc_plots(twc=actor, out_dir=videos_dir)

    report_path = write_report(metrics, model_path, reports_dir, cfg, video_path)

    print(
        f"Evaluation finished over {metrics['episodes']} episodes. "
        f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
    )
    print(f"Report saved to {report_path}")
    if video_path:
        print(f"Best episode video saved to {video_path}")


if __name__ == "__main__":
    main()
