"""Simulate a saved PyUriTwc policy, pick the best episode, and export videos.

The script searches the `solutions/solution_[A-C]` folders for the best TD3
actor checkpoint (``*_best_*.pth``), evaluates the requested solution to pick
the best seed, then replays all three checkpoints (initial, final, best) for
that seed and stitches them into a single side-by-side video (env + neuron
graph) played back-to-back.

Usage
-----
```bash
python scripts/sim_video.py solution_A --episodes 10 --video-dir out/videos
```

If the Micromamba environment is needed, activate it beforehand:
```
micromamba activate PyUri
```
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

# Use non-interactive backend to avoid display requirement when saving videos
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fiuri import build_fiuri_twc, build_fiuri_twc_v2  # noqa: E402
from td3_flat import TD3Config  # noqa: E402
from pyuri.model import TWC_JSON  # noqa: E402


ENV_ID = "MountainCarContinuous-v0"
BASE_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_solution_models(solution_dir: Path) -> Dict[str, Path]:
    """Return dict with first/initial, final, and best checkpoints if present."""
    ckpts = {
        "initial": None,
        "final": None,
        "best": None,
    }
    first = sorted(solution_dir.glob("*_first_*.pth"))
    if first:
        ckpts["initial"] = first[-1]
    final = sorted(solution_dir.glob("*_final_*.pth"))
    if final:
        ckpts["final"] = final[-1]
    best = sorted(solution_dir.glob("*_best_*.pth"))
    if best:
        ckpts["best"] = best[-1]
    return ckpts


def load_config(cfg_path: Path) -> TD3Config:
    cfg = TD3Config()
    with cfg_path.open("r") as f:
        cfg.load(json.load(f))
    return cfg


def build_actor(cfg: TD3Config) -> torch.nn.Module:
    """Instantiate the correct PyUriTwc variant based on config prefix."""
    use_sg = "SG" in cfg.model_prefix and "noSG" not in cfg.model_prefix
    actor = build_fiuri_twc_v2() if use_sg else build_fiuri_twc()
    actor.to(DEVICE)
    actor.eval()
    return actor


def load_actor_state(actor: torch.nn.Module, ckpt: Path) -> None:
    state = torch.load(ckpt, map_location=DEVICE)
    actor.load_state_dict(state)


@torch.no_grad()
def policy(actor: torch.nn.Module, obs: np.ndarray) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    action = actor(obs_t).squeeze(0).cpu().numpy()
    return action


def run_episode(
    env: gym.Env, actor: torch.nn.Module, seed: int, record_states: bool = False
) -> Tuple[float, int, Optional[np.ndarray]]:
    """Run one episode; optionally capture neuron internal states each step."""
    obs, _ = env.reset(seed=seed)
    if hasattr(actor, "reset"):
        actor.reset()

    done = False
    ep_reward = 0.0
    steps = 0
    states: List[np.ndarray] = []

    while not done:
        act = policy(actor, obs)
        # The forward call updates stored_E/ stored_O inside the actor
        if record_states and hasattr(actor, "stored_E") and actor.stored_E is not None:
            states.append(actor.stored_E.squeeze(0).detach().cpu().numpy())

        obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        ep_reward += reward
        steps += 1

    state_arr = np.stack(states, axis=0) if record_states and states else None
    return ep_reward, steps, state_arr


def evaluate_solution_best_seed(solution: str, episodes: int) -> Dict[str, float]:
    """Evaluate the best checkpoint of a single solution to pick the best seed."""
    sol_dir = ROOT / "solutions" / solution
    ckpts = list_solution_models(sol_dir)
    best_ckpt = ckpts.get("best")
    if best_ckpt is None:
        raise FileNotFoundError(f"No best checkpoint found in {sol_dir}")

    cfg = load_config(sol_dir / "full_config.json")
    actor = build_actor(cfg)
    load_actor_state(actor, best_ckpt)

    best = {
        "solution": solution,
        "reward": -np.inf,
        "seed": None,
        "ckpt": best_ckpt,
        "cfg": cfg,
    }

    env = gym.make(ENV_ID)
    for ep in range(episodes):
        seed = BASE_SEED + ep
        reward, steps, _ = run_episode(env, actor, seed)
        print(f"{solution} (best model) episode {ep} reward={reward:.3f} steps={steps}")
        if reward > best["reward"]:
            best.update({
                "reward": reward,
                "seed": seed,
            })
    env.close()
    return best


def compute_layout() -> Dict[str, Tuple[float, float]]:
    """Match the layered layout used in `twc_graph_gen.py` (horizontal bands, stacked vertically)."""
    positions: Dict[str, Tuple[float, float]] = {}
    layer_spec = {
        0: ["PVD", "PLM", "AVM", "ALM"],  # sensory / input
        1: ["DVA", "AVD", "PVC"],           # interneurons (stage 2)
        2: ["AVA", "AVB"],                   # interneurons (stage 3)
        3: ["REV", "FWD"],                   # motor / output
    }

    x_gap = 3.5
    y_gap = 2.2

    for li, names in layer_spec.items():
        x = li * x_gap
        y0 = (len(names) - 1) * y_gap / 2.0  # center nodes vertically in their layer
        for i, name in enumerate(names):
            positions[name] = (x, y0 - i * y_gap)
    return positions


def generate_combined_video(
    segments: List[Tuple[str, List[np.ndarray], np.ndarray, List[float]]],
    solution: str,
    output_path: Path,
    fps: int = 30,
):
    """Create a single video stitching multiple segments back-to-back."""
    # Normalize lengths per segment
    freeze_frames = max(1, int(fps * 0.5))  # hold last frame 0.5s

    trimmed_segments = []
    for label, frames, states, cum_rewards in segments:
        if states is None or states.ndim != 2 or not frames:
            continue
        T = min(len(frames), states.shape[0], len(cum_rewards))
        # Trim
        f = frames[:T]
        s = states[:T]
        r = cum_rewards[:T]
        # Freeze last frame
        f += [f[-1]] * freeze_frames
        s = np.concatenate([s, np.repeat(s[-1][None, :], freeze_frames, axis=0)], axis=0)
        r += [r[-1]] * freeze_frames
        trimmed_segments.append((label, f, s, r))

    if not trimmed_segments:
        raise ValueError("No valid segments to render")

    neuron_order = [name for name, _ in sorted(TWC_JSON["neurons"].items(), key=lambda x: x[1])]
    positions = compute_layout()
    xs = [positions[n][0] for n in neuron_order]
    ys = [positions[n][1] for n in neuron_order]

    # Precompute cumulative frame counts
    seg_lengths = [len(fr) for _, fr, _, _ in trimmed_segments]
    cum_lengths = np.cumsum([0] + seg_lengths)
    total_frames = cum_lengths[-1]

    fig, (ax_env, ax_neu) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(top=0.88)

    ax_env.axis("off")
    im_env = ax_env.imshow(trimmed_segments[0][1][0])
    reward_text = ax_env.text(
        0.02,
        0.95,
        "reward: 0.000",
        transform=ax_env.transAxes,
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
    )

    ax_neu.set_xlim(-1.5, 3 * 3.5 + 1.5)
    ax_neu.set_ylim(-3.8, 3.8)
    ax_neu.axis("off")

    edge_colors = {"EX": "forestgreen", "IN": "darkred", "GJ": "orange"}
    for edge in TWC_JSON["edges"]:
        src, dst, et = edge["src"], edge["dst"], edge["type"]
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        ax_neu.plot([x0, x1], [y0, y1], color=edge_colors.get(et, "gray"), alpha=0.4, linewidth=1.0)

    for name, (x, y) in positions.items():
        ax_neu.text(x, y + 0.08, name, ha="center", va="bottom", fontsize=9)

    scatter = ax_neu.scatter(
        xs,
        ys,
        c=trimmed_segments[0][2][0],
        cmap="RdYlGn",
        vmin=-10,
        vmax=10,
        s=420,
        edgecolors="black",
        linewidths=0.8,
    )

    title = ax_neu.set_title("Neuron states t=0", fontsize=10)
    banner = fig.suptitle(f"{solution} | {trimmed_segments[0][0]}", fontsize=13)

    def update(global_idx: int):
        # Determine which segment we're in
        seg_idx = np.searchsorted(cum_lengths[1:], global_idx, side="right")
        offset = global_idx - cum_lengths[seg_idx]
        label, frames, states, rewards = trimmed_segments[seg_idx]
        im_env.set_data(frames[offset])
        scatter.set_array(states[offset])
        title.set_text(f"Neuron states t={offset}")
        banner.set_text(f"Solution {solution} | {label}")
        reward_text.set_text(f"reward: {rewards[offset]:.3f}")
        return im_env, scatter, title, banner, reward_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / fps,
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)


def record_episode(
    actor: torch.nn.Module,
    seed: int,
) -> Tuple[List[np.ndarray], Optional[np.ndarray], float, List[float]]:
    """Roll out one episode and return env frames, neuron states, reward, and cumulative rewards."""
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    if hasattr(actor, "reset"):
        actor.reset()

    done = False
    states: List[np.ndarray] = []
    frames: List[np.ndarray] = []
    reward_sum = 0.0
    cum_rewards: List[float] = []

    while not done:
        act = policy(actor, obs)
        if hasattr(actor, "stored_E") and actor.stored_E is not None:
            states.append(actor.stored_E.squeeze(0).detach().cpu().numpy())

        obs, reward, terminated, truncated, _ = env.step(act)
        frame = env.render()
        frames.append(frame)
        done = terminated or truncated
        reward_sum += reward
        cum_rewards.append(reward_sum)

    env.close()
    states_arr = np.stack(states, axis=0) if states else None
    return frames, states_arr, reward_sum, cum_rewards


def main():
    parser = argparse.ArgumentParser(description="Simulate a solution's PyUriTwc checkpoints and export a combined video")
    parser.add_argument("solution", help="Solution folder name under ./solutions (e.g., solution_A)")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes (with different seeds) used to pick best seed")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "out" / "videos")
    parser.add_argument("--fps", type=int, default=30, help="FPS for neuron animation")
    args = parser.parse_args()

    best = evaluate_solution_best_seed(args.solution, args.episodes)
    print(
        f"Best seed for {best['solution']} based on BEST checkpoint: seed={best['seed']} reward={best['reward']:.3f}"
    )

    ckpts = list_solution_models(ROOT / "solutions" / best["solution"])
    order = [
        ("initial", ckpts.get("initial")),
        ("final", ckpts.get("final")),
        ("best", ckpts.get("best")),
    ]

    segments = []
    for label, ckpt in order:
        if ckpt is None:
            print(f"Checkpoint for {label} not found; skipping")
            continue
        actor = build_actor(best["cfg"])
        load_actor_state(actor, ckpt)
        frames, states, rew, cum_rewards = record_episode(actor, best["seed"])
        print(f"{label} reward: {rew:.3f} | frames: {len(frames)} | states: {None if states is None else states.shape[0]}")
        segments.append((label, frames, states, cum_rewards))

    if not segments:
        print("No segments recorded; nothing to render.")
        return

    args.video_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{best['solution']}_seed{best['seed']}"
    combined_path = args.video_dir / f"{tag}_all-checkpoints.mp4"
    generate_combined_video(segments, best["solution"], combined_path, fps=args.fps)
    print(f"Combined env + neuron video saved to: {combined_path}")


if __name__ == "__main__":
    main()
