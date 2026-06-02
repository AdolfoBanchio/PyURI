import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import json
import math
import numpy as np
import torch
import optuna
from datetime import datetime
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from torch.utils.tensorboard import SummaryWriter

from fiuri import PyUriTwc_V2, TWC_JSON
from ppo.ppo import PPOConfig, PPOEngine, ppo_train
from mlp import ValueCriticInvPen
from twc import twc_io as io


def make_env(seed: int, env_id: str = "InvertedPendulum-v5") -> object:
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def phi_invpen_gentle(obs) -> float:
    # Soft Gaussian potential on pole angle only. Max 0.1 at upright,
    # decays to ~0 at termination boundary. Shaping deltas stay ~0.01,
    # so PPO sees mostly the +1/step base reward.
    angle = obs[1]
    return 0.1 * math.exp(-(angle / 0.15) ** 2)


def ppo_invpen_score(all_returns: list[float], all_steps: list[float]) -> float:
    # Variance penalty loosened from 0.6 → 0.3 so high-peak / noisy-plateau
    # trials (e.g. trial 54 reached 102 peak but oscillated ±6 at the end)
    # aren't over-penalised vs. stable-but-mediocre ones.
    window_ret = np.array(all_returns[-6:], dtype=np.float64)
    mean_ret   = float(window_ret.mean())
    std_ret    = float(window_ret.std()) if len(window_ret) > 1 else 0.0
    return mean_ret - 0.3 * std_ret


# Floor below which the untrained TWC's obs-conditional output is too weak
# for PPO to ever produce an obs-responsive policy. Trials hitting this
# are pruned immediately to save compute.
DEAD_INIT_RAW_STD_FLOOR = 0.03


# PPO trial 54 (from study twc_invpen_ppo_SG) — the breakthrough run.
# Enqueued as a sanity-check replay under the new settings. The seed below
# matches the original run's deterministic formula (54 * 1000 + 1 = 54001),
# which is what produced the obs-responsive TWC init (raw_std ≈ 0.151).
TRIAL_54_PARAMS = {
    "episodes_per_update": 14,
    "ppo_epochs":          10,
    "log_std_init":       -0.2339073901018924,
    "actor_lr":            0.0008814454723153739,
    "critic_lr":           0.0007475491972398553,
    "calib_target_std":    2.3594809937508354,
    "steepness_fire":      9.946721125616367,
    "steepness_gj":       10.47776186246416,
    "steepness_input":     3.4791684366144686,
    "input_thresh":        0.008420218586737102,
}
TRIAL_54_SEED = 54001


def objective(trial: optuna.Trial, study_name: str) -> float:
    cfg = PPOConfig()
    cfg.device          = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.max_train_steps = 700_000
    cfg.eval_episodes   = 50
    cfg.model_prefix    = "ppo_SG_actor_invpen"

    cfg.mini_batch_size       = 256
    cfg.max_grad_norm         = 0.5
    # v4 raised log_std_min to -1.5 to keep exploration; the v4 replay of
    # trial 54 showed this prevented the policy from committing to its
    # learned competence. Reverted to -3.0.
    cfg.log_std_min           = -3.0
    cfg.log_std_max           =  0.5
    cfg.gamma                 = 0.99
    cfg.lam                   = 0.95
    cfg.clip_eps              = 0.2
    # v4 raised entropy_coef to 0.015; same regression as log_std_min.
    cfg.entropy_coef          = 0.005
    cfg.eval_interval_updates = 5
    cfg.burn_in_length        = 2
    # v4 raised calib_max_scale to 8.0; this gave the head excessive
    # leverage and destabilised trial 54's replay (peak 59 vs original
    # 102). Reverted to 5.0.
    cfg.calib_max_scale       = 5.0

    cfg.episodes_per_update = trial.suggest_int("episodes_per_update", 8, 24)
    cfg.ppo_epochs          = trial.suggest_int("ppo_epochs", 8, 15)
    # log_std_init range kept wider than the original [-0.7, -0.2]: the v4
    # winner (trial 5) used log_std_init=+0.004, only reachable in the
    # wider range. Keeping this is the one genuinely useful v4 finding.
    cfg.log_std_init        = trial.suggest_float("log_std_init", -0.4, 0.1)

    # actor_lr cap raised 1e-3 → 1.5e-3 (top trials were pinned at the
    # previous upper edge).
    cfg.actor_lr  = trial.suggest_float("actor_lr",  5e-4, 1.5e-3, log=True)
    cfg.critic_lr = trial.suggest_float("critic_lr", 3e-4, 2e-3,   log=True)

    cfg.calib_target_std = trial.suggest_float("calib_target_std", 1.5, 2.5)

    # TWC-specific. internal_steps controls how many times _physics_step is
    # unrolled per env step so signal can propagate from input to output
    # neurons before the action is decoded. Narrowed to {3, 4} based on the
    # previous study's top-trial cluster.
    internal_steps = 3

    # Surrogate-gradient gate sharpness. steepness_fire/gj control how
    # closely the firing-gate sigmoid / gap-junction tanh approximate the
    # hard threshold/sign in V1. steepness_input controls the decay-gate
    # softness. input_thresh is the influence magnitude below which a
    # neuron is treated as "no input" → decay path.
    steepness_fire  = trial.suggest_float("steepness_fire",  8.0, 14.0)
    steepness_gj    = trial.suggest_float("steepness_gj",    6.0, 12.0)
    steepness_input = trial.suggest_float("steepness_input", 3.0,  6.0)
    input_thresh    = trial.suggest_float("input_thresh",    0.005, 0.05, log=True)

    # Deterministic per-trial seed. The trial 54 replay needs its original
    # seed (54001) so the TWC init reproduces the breakthrough raw_std; for
    # all other trials, derive from trial.number.
    if trial.params == TRIAL_54_PARAMS:
        seed = TRIAL_54_SEED
    else:
        seed = trial.number * 1000 + 1
    cfg.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)

    state_dim = env.observation_space.shape[0]
    device    = torch.device(cfg.device)

    actor = PyUriTwc_V2(
        config_json     = TWC_JSON,
        obs_encoder     = io.ipen_obs_to_potentials_v2,
        action_decoder  = io.twc_out_2_invpen_mean,
        internal_steps  = internal_steps,
        steepness_gj    = steepness_gj,
        steepness_fire  = steepness_fire,
        steepness_input = steepness_input,
        input_thresh    = input_thresh,
    )
    critic     = ValueCriticInvPen(state_dim=state_dim)
    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    actor.to(device)
    critic.to(device)

    engine = PPOEngine(
        actor            = actor,
        critic           = critic,
        action_space     = env.action_space,
        actor_optimizer  = actor_opt,
        critic_optimizer = critic_opt,
        config           = cfg,
        device           = device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"trial_{trial.number}_{timestamp}"
    log_dir   = f"out/runs/optuna/{study_name}/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer    = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, "trial_params.json"), "w") as f:
        json.dump(trial.params, f, indent=4)
    with open(os.path.join(log_dir, "full_config.json"), "w") as f:
        f.write(cfg.to_json())

    # Calibrate once up-front so we can gate on raw_std before spending the
    # full training budget on a dead-init TWC.
    calib = engine.calibrate_output_head(env, target_std=cfg.calib_target_std)
    raw_std = calib["calib/raw_std"]
    trial.set_user_attr("calib_raw_std", raw_std)
    for k, v in calib.items():
        writer.add_scalar(k, v, 0)
    if raw_std < DEAD_INIT_RAW_STD_FLOOR:
        env.close(); writer.close()
        return -50.0

    try:
        all_eval_returns, all_eval_steps, best_model_path = ppo_train(
            env=env,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=trial,
            phi=phi_invpen_gentle,
            skip_calibration=True,
        )

        if len(all_eval_returns) < 2:
            return -200.0
        if np.mean(all_eval_returns[-6:]) < 8.0:
            return -100.0
        return ppo_invpen_score(all_eval_returns, all_eval_steps)

    except optuna.TrialPruned:
        return -200.0
    finally:
        env.close()
        writer.close()


def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = "twc_invpen_ppo_SG_v5"

    sampler = TPESampler(multivariate=True, group=True, n_startup_trials=8)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=4,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///twc_ppo_invpen_SG_v5.sqlite3",
        study_name=study_name,
        load_if_exists=True,
    )

    # Sanity-check replay of PPO trial 54 under the new settings (see
    # TRIAL_54_PARAMS at the top of the file; the seed override that pairs
    # with it lives in objective()).
    if not any(t.params == TRIAL_54_PARAMS for t in study.trials):
        study.enqueue_trial(TRIAL_54_PARAMS)

    obj_wrapper = lambda trial: objective(trial, study_name)

    try:
        study.optimize(obj_wrapper, n_trials=40, gc_after_trial=True, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Optuna study interrupted by user.")

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed_trials:
        print(f"\nStudy '{study_name}' complete.")
        print(f"Best score: {study.best_value:.2f}")
        print("Best params:")
        print(json.dumps(study.best_params, indent=4))

        out_dir = f"out/runs/optuna/{study_name}_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "best_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=4)
    else:
        print(f"\nStudy '{study_name}' has no completed trials yet.")


if __name__ == "__main__":
    main()
