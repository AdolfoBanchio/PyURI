import argparse
import pickle
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ariel import ModelInterfaces as mi


class ArielPickleUnpickler(pickle.Unpickler):
    MODULE_MAP = {
        "Models.Fiuri.Model": "ariel.Model",
        "Models.Fiuri.NeuralNetwork": "ariel.NeuralNetwork",
        "Models.Fiuri.Neuron": "ariel.Neuron",
        "Models.Fiuri.Connection": "ariel.Connection",
        "Models.Fiuri.ModelInterfaces": "ariel.ModelInterfaces",
    }

    def find_class(self, module, name):
        module = self.MODULE_MAP.get(module, module)
        return super().find_class(module, name)


def load_ariel_model_from_pickle(pickle_path: Path):
    with pickle_path.open("rb") as handle:
        return ArielPickleUnpickler(handle).load()


def run_episode(env: gym.Env, model, seed: int):
    obs, _ = env.reset(seed=seed)
    model.Reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        model_obs = mi.envObsToModelObs(obs)
        action_val = model.Update(model_obs)
        action = mi.modActionToEnvAction(action_val)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    return total_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="src/ariel/model/model",
        help="Path to the pickled ariel model dump.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = load_ariel_model_from_pickle(Path(args.model_path))
    env = gym.make("MountainCarContinuous-v0")

    rewards = []
    steps = []
    for ep in range(args.episodes):
        reward, n_steps = run_episode(env, model, args.seed + ep)
        rewards.append(reward)
        steps.append(n_steps)

    env.close()

    rewards_np = np.array(rewards, dtype=np.float32)
    steps_np = np.array(steps, dtype=np.int32)
    print(f"episodes={args.episodes}")
    print(f"mean_reward={rewards_np.mean():.4f}")
    print(f"std_reward={rewards_np.std(ddof=1) if args.episodes > 1 else 0.0:.4f}")
    print(f"mean_steps={steps_np.mean():.2f}")
    print(f"min_steps={steps_np.min()}")
    print(f"max_steps={steps_np.max()}")


if __name__ == "__main__":
    main()
