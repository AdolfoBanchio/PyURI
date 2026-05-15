import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import json
import gymnasium as gym
import numpy as np
import torch
import optuna
from dataclasses import dataclass, asdict
from datetime import datetime
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import OUNoise, SequenceBuffer
from mlp import TwinCritic
from fiuri import build_fiuri_twc_mcc
from td3_flat import TD3Engine, TD3Config, td3_train
import fcntl

def make_env(seed, env_id="MountainCarContinuous-v0"):
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def bootstrap_enqueue_once(study: optuna.Study, best_configs, best_seeds, lock_path: str):
    """
    Encola los trials iniciales UNA sola vez, incluso si lanzás múltiples procesos.
    Usa un lock de archivo + marca en system_attrs.
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        # Re-check ya con lock tomado
        sys_attrs = study._storage.get_study_system_attrs(study._study_id)
        if sys_attrs.get("bootstrap_done", False):
            return

        # Encolar configs + seed
        for cfg_dict, seed in zip(best_configs, best_seeds):
            cfg_with_seed = dict(cfg_dict)
            cfg_with_seed["seed"] = int(seed)
            study.enqueue_trial(cfg_with_seed)

        # Marcar como hecho (queda persistido en la DB)
        study._storage.set_study_system_attr(study._study_id, "bootstrap_done", True)

        fcntl.flock(f, fcntl.LOCK_UN)

def phi_mcc(obs):
    # Potential shaping (pos only). Keeps your original behavior.
    pos = obs[0]
    pos_min, pos_max = -1.2, 0.6
    x = (pos - pos_min) / (pos_max - pos_min)
    return 4 * float(np.clip(x, 0.0, 1.0))

def mcc_score_fn(m_ret, s_rate, s_suc, m_act, all_scores, all_steps):
    # Calculamos las medias de la ventana (6)
        window_rew = np.mean(all_scores[-6:])
        # Si avg_steps es inf (no llegó), le asignamos el máximo del entorno (1000)
        clean_steps = [s if np.isfinite(s) else 1000 for s in all_steps[-6:]]
        window_steps = np.mean(clean_steps)

        step_penalty = max(0, (window_steps - 200) * 0.1)
        current_combined_score = window_rew - step_penalty

        return current_combined_score

def objective(trial: optuna.Trial, study_name: str):
    cfg = TD3Config()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Shorter runs for search; keep evaluation cadence stable
    cfg.max_train_steps = 200_000 
    cfg.warmup_steps = 8_000  # Empezamos a aprender un poco antes
    cfg.eval_interval_episodes = 10
    cfg.eval_episodes = 10
    cfg.num_update_loops = 1
    cfg.update_every = 1
    cfg.batch_size = 256
    cfg.model_prefix = "td3_flat_actor_noSG"

    # --- Tunable Hyperparameters ---
    cfg.sequence_length = trial.suggest_categorical("sequence_length", [8, 12, 16])
    cfg.burn_in_length = cfg.sequence_length // 2

    cfg.actor_lr = trial.suggest_float("actor_lr", 1.5e-4, 5e-4, log=True)
    cfg.critic_lr = trial.suggest_float("critic_lr", 1.5e-4, 5e-4, log=True)
    
    cfg.gamma = trial.suggest_float("gamma", 0.96, 0.992)    
    cfg.tau = trial.suggest_float("tau", 0.005, 0.01)
    
    cfg.target_noise = trial.suggest_float("target_noise", 0.20, 0.35)
    cfg.noise_clip = trial.suggest_float("noise_clip", 0.30, 0.45)
    
    cfg.ou_sigma_init = trial.suggest_float("sigma_start", 0.35, 0.50)
    cfg.ou_sigma_end = trial.suggest_float("sigma_end", 0.01, 0.10)

    cfg.ou_sigma_decay_steps = int(cfg.max_train_steps * 0.60)

    seed = trial.suggest_int("seed", 0, 2_000_000_000)
    cfg.seed = seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = build_fiuri_twc()
    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    actor.to(cfg.device)
    critic.to(cfg.device)
    engine = TD3Engine(
        gamma=cfg.gamma,
        tau=cfg.tau,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        policy_delay=cfg.policy_delay,
        target_policy_noise=cfg.target_noise,
        target_noise_clip=cfg.noise_clip,
        device=cfg.device,
    )

    replay_buf = SequenceBuffer(capacity=cfg.replay_buffer_size, device=cfg.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"trial_{trial.number}_{timestamp}"
    log_dir = f"out/runs/optuna/{study_name}/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, "trial_params.json"), "w") as f:
        json.dump(trial.params, f, indent=4)

    with open(os.path.join(log_dir, "full_config.json"), "w") as f:
        f.write(cfg.to_json())

    try:
        all_eval_scores, all_eval_steps,best_model_path =     td3_train(
            env=env,
            replay_buf=replay_buf,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=trial,
            phi=phi_mcc,
            model_score_fn=mcc_score_fn,
            log_interval=200
        )

        # Tomamos los últimos 6 resultados de estabilidad
        final_rew = np.mean(all_eval_scores[-6:])
        
        # Procesamos los pasos: convertimos inf a 1000 para promediar
        final_steps_raw = [s if np.isfinite(s) else 1000 for s in all_eval_steps[-6:]]
        final_avg_steps = np.mean(final_steps_raw)

        # MÉTRICA DE ÉXITO DE ALTA EFICIENCIA
        # Queremos que un Retorno de 95 con 120 pasos sea MEJOR que 
        # un Retorno de 98 con 400 pasos.
        
        # Penalización: 0.2 por paso arriba de 100 (más agresiva al final)
        penalty = max(0, (final_avg_steps - 200) * 0.1)
        
        # Si el agente nunca llegó a la meta consistentemente, el score cae drásticamente
        if final_rew < 50:
            return -100 # Zona de fracaso
            
        return final_rew - penalty
    
    except optuna.TrialPruned:
        return -200.0
    finally:
        env.close()
        writer.close()

best_configs_seeds = [44, 57, 63, 82, 85, 88]
best_configs = [
    {
        "sequence_length": 12,
        "actor_lr": 0.00024282916261759212,
        "critic_lr": 0.0003969776718059757,
        "gamma": 0.9806389849437517,
        "tau": 0.009996203689469246,
        "target_noise": 0.3289145534111374,
        "noise_clip": 0.38897743928200934,
        "sigma_start": 0.39665599334653406,
        "sigma_end": 0.031213657895530607,
        "seed": 44
    },
    {
        "sequence_length": 12,
        "actor_lr": 0.0004371390087077319,
        "critic_lr": 0.0004917831328589752,
        "gamma": 0.9723323855552827,
        "tau": 0.007353818968043093,
        "target_noise": 0.3054387337757573,
        "noise_clip": 0.40108735608811635,
        "sigma_start": 0.38906086078700075,
        "sigma_end": 0.03634835547945163,
        "seed": 57
    },
    {
        "sequence_length": 16,
        "actor_lr": 0.00020539252385050292,
        "critic_lr": 0.00044095064414419806,
        "gamma": 0.9842885185148715,
        "tau": 0.007951845679898808,
        "target_noise": 0.3264477882187622,
        "noise_clip": 0.3340577908751693,
        "sigma_start": 0.3724539295678623,
        "sigma_end": 0.022336233050158937,
        "seed": 63
    },
    {
        "sequence_length": 12,
        "actor_lr": 0.00024059232431064746,
        "critic_lr": 0.0003624535315024231,
        "gamma": 0.9831198615149824,
        "tau": 0.00997857171214146,
        "target_noise": 0.3054206984013033,
        "noise_clip": 0.41486226082484023,
        "sigma_start": 0.3941061809470922,
        "sigma_end": 0.023129421806653883,
        "seed": 82
    },
    {
        "sequence_length": 12,
        "actor_lr": 0.00028573633030379487,
        "critic_lr": 0.00044567904730435915,
        "gamma": 0.9755559775062949,
        "tau": 0.008625177204637152,
        "target_noise": 0.2908854735252029,
        "noise_clip": 0.3459687446093757,
        "sigma_start": 0.391109193636304,
        "sigma_end": 0.037051657717669585,
        "seed": 85
    },
    {
        "sequence_length": 16,
        "actor_lr": 0.00015106041248486588,
        "critic_lr": 0.00022054545447943227,
        "gamma": 0.9812235517421997,
        "tau": 0.006077770102125573,
        "target_noise": 0.31001692997625235,
        "noise_clip": 0.37865967816481977,
        "sigma_start": 0.3966796152074734,
        "sigma_end": 0.0722901509347865,
        "seed": 88
    }
]

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = "twc_mcc_td3_flat_noSG_comb_score"

    sampler = TPESampler(
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        storage="sqlite:///td3_comb_score.sqlite3",
        study_name=study_name,
        load_if_exists=True,
    )


    lock_path = f"out/{study_name}.lock"
    bootstrap_enqueue_once(study, best_configs, best_configs_seeds, lock_path)

    obj_wrapper = lambda trial: objective(trial, study_name)
    
    try:
        study.optimize(
            obj_wrapper,
            n_trials=20,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Optuna study interrupted by user.")

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed_trials:
        print(f"\nStudy '{study_name}' complete.")
        print(f"Best value (max eval return): {study.best_value:.2f}")
        print("Best params:")
        print(json.dumps(study.best_params, indent=4))

        out_dir = f"out/runs/optuna/{study_name}_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)
        best_params_path = os.path.join(out_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        print(f"Saved to {best_params_path}")
    else:
        print(f"\nStudy '{study_name}' has no completed trials yet (still running in other workers?).")


if __name__ == "__main__":
    main()
