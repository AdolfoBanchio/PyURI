import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlp import Actor
from twc import mcc_obs_encoder, twc_out_2_mcc_action, build_twc

ENV = "MountainCarContinuous-v0"
SEED = 42
DEFAULT_EPISODES = 1000
DEFAULT_ACTOR_HIDDEN = [400, 300]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_actor(model_path: str, env: gym.Env, hidden_sizes=None) -> torch.nn.Module:
    """Create the right actor type and load the provided weights."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if "twc" in model_path.name.lower():
        actor = build_twc(
            obs_encoder=mcc_obs_encoder, action_decoder=twc_out_2_mcc_action, log_stats=False, internal_steps=3
        )
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        hidden_sizes = hidden_sizes or DEFAULT_ACTOR_HIDDEN
        actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=float(env.action_space.high[0]),
            size=hidden_sizes,
        )

    state_dict = torch.load(model_path, map_location=DEVICE)
    print(state_dict)
    actor.load_state_dict(state_dict)
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

    record_env.close()
    return getattr(record_env.video_folder, "path", None)


def write_report(
    metrics: dict,
    model_path: Path,
    reports_dir: Path,
    video_path: Optional[str],
):
    """Persist evaluation summary to CSV named after the model checkpoint."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_path.stem
    report_path = reports_dir / f"{model_name}.csv"

    fields = [
        "timestamp",
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
        "timestamp": datetime.utcnow().isoformat(),
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

    with report_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a MountainCarContinuous policy and export metrics."
    )
    parser.add_argument("model_path", type=str, help="Path to the actor state_dict to evaluate.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of evaluation episodes (default: {DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory where the CSV report will be stored.",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="videos",
        help="Directory where evaluation videos will be stored.",
    )
    parser.add_argument(
        "--record-best",
        action="store_true",
        default=True,
        help="If set, record the best episode based on total reward.",
    )
    parser.add_argument(
        "--mlp-hidden",
        type=list[int],
        nargs="+",
        help="Optional hidden sizes for the MLP actor (ignored for TWC models).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    videos_dir = Path(args.videos_dir)
    model_path = Path(args.model_path)

    env = gym.make(ENV)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    actor = get_actor(model_path, env, hidden_sizes=args.mlp_hidden)
    metrics = evaluate_model(env, actor, n_eps=args.episodes)
    env.close()

    video_path = None
    if args.record_best:
        video_path = record_episode(model_path.stem, actor, metrics["best_seed"], videos_dir)

    report_path = write_report(metrics, model_path, reports_dir, video_path)

    print(
        f"Evaluation finished over {metrics['episodes']} episodes. "
        f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
    )
    print(f"Report saved to {report_path}")
    if video_path:
        print(f"Best episode video saved to {video_path}")


if __name__ == "__main__":
    main()
