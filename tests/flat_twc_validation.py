import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import random
import matplotlib.pyplot as plt
import torch
import torch.functional as F
import numpy as np
import gymnasium as gym
from twc.twc_io import (
    mcc_obs_encoder,
    twc_out_2_mcc_action,
)
from ariel.Model import Model as FiuModel
from ariel import Connection as con
from fiuri import build_fiuri_twc_v2 as build_fiuri_twc


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # (opcional) determinismo; puede bajar perf, pero para test está ok
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def run_one_episode(env, actor, device, max_steps=1000):
    obs, _ = env.reset()
    actor.reset(1)  # importante: reset SOLO al inicio
    fwd_idx = actor.neuron_names['FWD']
    rev_idx = actor.neuron_names['REV']

    actions = []
    diffs = []  # E_FWD - E_REV
    fwd_gt = 0
    rev_gt = 0
    sat = 0
    pos = 0
    neg = 0
    ret = 0.0
    
    for t in range(max_steps):
        obs_t = torch.tensor([obs], dtype=torch.float32, device=device)
        a = actor(obs_t).item()   # asume que devuelve escalar
        # clip defensivo por si algo se va de rango numérico
        a = float(np.clip(a, -1.0, 1.0))

        # stats acción
        actions.append(a)
        if a > 0: pos += 1
        elif a < 0: neg += 1
        if abs(a) > 0.9: sat += 1

        # stats polarización interna
        E_fwd = actor.stored_E[0, fwd_idx].item()
        E_rev = actor.stored_E[0, rev_idx].item()
        d = E_fwd - E_rev
        diffs.append(d)
        if d > 0: fwd_gt += 1
        elif d < 0: rev_gt += 1

        obs, reward, terminated, truncated, _ = env.step([a])
        ret += float(reward)
        if terminated or truncated:
            break

    T = len(actions)
    actions = np.array(actions, dtype=np.float32)
    diffs = np.array(diffs, dtype=np.float32)

    return {
        "T": T,
        "return": ret,
        "mean_action": float(actions.mean()),
        "std_action": float(actions.std()),
        "pct_action_pos": pos / T,
        "pct_action_neg": neg / T,
        "pct_action_sat": sat / T,
        "mean_Ediff": float(diffs.mean()),
        "std_Ediff": float(diffs.std()),
        "pct_FWD_gt_REV": fwd_gt / T,
        "pct_REV_gt_FWD": rev_gt / T,
        "success": bool(terminated)  # en MCC, terminated suele ser llegar a la bandera
    }

def polarization_sweep(
    seeds=range(10),
    max_steps=1000,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("MountainCarContinuous-v0")
    results = []

    for seed in seeds:
        set_all_seeds(seed)

        # IMPORTANTE: re-crear actor por seed (pesos random distintos)
        actor = build_fiuri_twc()
        actor.to(device=device)
        actor.device = device
        actor.reset(1)
        # correr episodio
        stats = run_one_episode(env, actor, device=device, max_steps=max_steps)
        stats["seed"] = int(seed)
        results.append(stats)

        print(
            f"[seed {seed:2d}] T={stats['T']:4d} "
            f"R={stats['return']:+7.2f} "
            f"succ={stats['success']} "
            f"pct_pos={stats['pct_action_pos']:.2f} "
            f"pct_sat={stats['pct_action_sat']:.2f} "
            f"pct_FWDgt={stats['pct_FWD_gt_REV']:.2f}"
        )

    env.close()
    return results

if __name__ == "__main__":
    results = polarization_sweep(seeds=range(10), max_steps=1000)

    # Resumen final
    returns = [r["return"] for r in results]
    succs = [r["success"] for r in results]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    succ_rate = np.mean(succs)

    print("\n=== Summary ===")
    print(f"Mean Return: {mean_return:+7.2f} ± {std_return:.2f}")
    print(f"Success Rate: {succ_rate*100:.1f}% ({sum(succs)}/{len(succs)})")