"""Simulate a saved PyUriTwc policy, pick the best episode, and export videos.

The script searches the `solutions/solution_[A-C]` folders for the best TD3
actor checkpoint (``*_best_*.pth``), evaluates each model for a small number of
episodes, keeps the episode with the highest return, and then:

1) Records the environment roll-out as an mp4.
2) Records the neuron-state dynamics as a node-graph animation (red -> green
   for state values in [-10, 10]).

Usage
-----
```bash
python scripts/sim_video.py --episodes 10 --video-dir out/videos
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
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

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


def list_solution_models(solution_dir: Path) -> List[Path]:
    """Return sorted list of best checkpoints inside a solution folder."""
    return sorted(solution_dir.glob("*_best_*.pth"))


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


def evaluate_solutions(
    solutions: Iterable[str], episodes: int
) -> Dict[str, float]:
    """Evaluate each solution for ``episodes`` and return best episode info."""
    best = {
        "solution": None,
        "reward": -np.inf,
        "seed": None,
        "ckpt": None,
        "cfg": None,
    }

    for sol in solutions:
        sol_dir = ROOT / "solutions" / sol
        ckpts = list_solution_models(sol_dir)
        if not ckpts:
            print(f"No best checkpoint found in {sol_dir}, skipping")
            continue

        cfg = load_config(sol_dir / "full_config.json")
        actor = build_actor(cfg)
        load_actor_state(actor, ckpts[-1])  # latest best

        env = gym.make(ENV_ID)
        for ep in range(episodes):
            seed = BASE_SEED + ep
            reward, steps, _ = run_episode(env, actor, seed)
            print(f"{sol} episode {ep} reward={reward:.3f} steps={steps}")
            if reward > best["reward"]:
                best.update({
                    "solution": sol,
                    "reward": reward,
                    "seed": seed,
                    "ckpt": ckpts[-1],
                    "cfg": cfg,
                })
        env.close()

    return best


def compute_layout() -> Dict[str, Tuple[float, float]]:
    """Layered layout with four horizontal bands (sensory → interneuron-1 → interneuron-2 → motor)."""
    positions: Dict[str, Tuple[float, float]] = {}
    layer_spec = {
        0: ["PVD", "PLM", "AVM", "ALM"],  # sensory / input
        1: ["DVA", "AVD", "PVC"],           # interneurons stage 2
        2: ["AVA", "AVB"],                   # interneurons stage 3
        3: ["REV", "FWD"],                   # motor / output
    }

    for li, names in layer_spec.items():
        y = float(li)
        xs = np.linspace(-1.0, 1.0, len(names)) if len(names) > 1 else np.array([0.0])
        for x, name in zip(xs, names):
            positions[name] = (float(x), y)
    return positions


def generate_neuron_video(
    states: np.ndarray,
    output_path: Path,
    fps: int = 30,
):
    """Save a node-graph animation of neuron internal states to ``output_path``."""
    if states.ndim != 2:
        raise ValueError("states array must be (T, num_neurons)")

    neuron_order = [name for name, _ in sorted(TWC_JSON["neurons"].items(), key=lambda x: x[1])]
    positions = compute_layout()

    xs = [positions[n][0] for n in neuron_order]
    ys = [positions[n][1] for n in neuron_order]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 3.3)
    ax.axis("off")

    # Draw static edges grouped by type for a bit of structure
    edge_colors = {"EX": "forestgreen", "IN": "darkred", "GJ": "orange"}
    for edge in TWC_JSON["edges"]:
        src, dst, et = edge["src"], edge["dst"], edge["type"]
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        ax.plot([x0, x1], [y0, y1], color=edge_colors.get(et, "gray"), alpha=0.4, linewidth=1.0)

    for name, (x, y) in positions.items():
        ax.text(x, y + 0.08, name, ha="center", va="bottom", fontsize=9)

    scatter = ax.scatter(
        xs,
        ys,
        c=states[0],
        cmap="RdYlGn",
        vmin=-10,
        vmax=10,
        s=420,
        edgecolors="black",
        linewidths=0.8,
    )

    title = ax.set_title("t = 0", fontsize=10)

    def init():
        scatter.set_array(states[0])
        title.set_text("t = 0")
        return scatter, title

    def update(frame: int):
        scatter.set_array(states[frame])
        title.set_text(f"t = {frame}")
        return scatter, title

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=states.shape[0],
        interval=1000 / fps,
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)


def record_best_episode(
    actor: torch.nn.Module,
    seed: int,
    video_dir: Path,
    tag: str,
) -> Tuple[Optional[str], Optional[np.ndarray], float]:
    """Record env video and collect neuron states for a single episode."""
    base_env = gym.make(ENV_ID, render_mode="rgb_array")
    record_env = RecordVideo(
        base_env,
        video_folder=str(video_dir),
        episode_trigger=lambda idx: True,
        name_prefix=tag,
    )

    obs, _ = record_env.reset(seed=seed)
    if hasattr(actor, "reset"):
        actor.reset()

    done = False
    states: List[np.ndarray] = []
    reward_sum = 0.0

    while not done:
        act = policy(actor, obs)
        if hasattr(actor, "stored_E") and actor.stored_E is not None:
            states.append(actor.stored_E.squeeze(0).detach().cpu().numpy())
        obs, reward, terminated, truncated, _ = record_env.step(act)
        done = terminated or truncated
        reward_sum += reward

    video_name = getattr(record_env, "_video_name", None)
    video_path = str(video_dir / f"{video_name}.mp4") if video_name else None
    record_env.close()

    states_arr = np.stack(states, axis=0) if states else None
    return video_path, states_arr, reward_sum


def main():
    parser = argparse.ArgumentParser(description="Simulate best PyUriTwc model and export videos")
    parser.add_argument(
        "--solutions",
        nargs="+",
        default=["solution_A", "solution_B", "solution_C"],
        help="Solution folders to search under ./solutions",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per solution for selection")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "out" / "videos")
    parser.add_argument("--fps", type=int, default=30, help="FPS for neuron animation")
    args = parser.parse_args()

    best = evaluate_solutions(args.solutions, args.episodes)
    if best["solution"] is None:
        print("No valid model found in provided solutions.")
        return

    print(
        f"Best episode from {best['solution']} seed={best['seed']} reward={best['reward']:.3f}"
    )

    # Reload best actor cleanly
    best_actor = build_actor(best["cfg"])
    load_actor_state(best_actor, best["ckpt"])

    args.video_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{best['solution']}_seed{best['seed']}"
    env_video_path, states, rerun_reward = record_best_episode(
        best_actor, best["seed"], args.video_dir, tag
    )

    print(f"Rerun reward (should match selection): {rerun_reward:.3f}")
    if env_video_path:
        print(f"Environment video saved to: {env_video_path}")

    if states is not None:
        neuron_video_path = args.video_dir / f"{tag}_neurons.mp4"
        generate_neuron_video(states, neuron_video_path, fps=args.fps)
        print(f"Neuron dynamics video saved to: {neuron_video_path}")
    else:
        print("No neuron states captured; skipping neuron video.")


if __name__ == "__main__":
    main()
