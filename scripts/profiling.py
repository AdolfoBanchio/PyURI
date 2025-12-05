import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import os
import itertools
import json
import argparse
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from td3 import TD3Engine, TD3Config, td3_train
from utils import OUNoise, SequenceBuffer    
from mlp import Critic
from twc import (
    build_twc,
    mcc_obs_encoder,
    twc_out_2_mcc_action,
)

# --- Profiling imports ---
import cProfile
import pstats
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


def make_env(seed, env_id="MountainCarContinuous-v0"):
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MCC agent using TD3+TWC with optional CPU/GPU profiling"
    )
    parser.add_argument("config_path", type=str, help="Path to the TD3 Config json")

    # Flags de profiling
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Enable torch.profiler (CPU+CUDA ops profiling)",
    )
    parser.add_argument(
        "--cpu-profiler",
        action="store_true",
        help="Enable cProfile (Python-level profiling)",
    )
    parser.add_argument(
        "--profile-max-train-steps",
        type=int,
        default=None,
        help="Override cfg.max_train_steps ONLY for this run (useful for profiling short runs)",
    )

    return parser.parse_args()


def build_training_objects(cfg: TD3Config):
    # --- Semillas ---
    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- Actor: TWC + Fiuri ---
    v2_params = {}
    if getattr(cfg, "use_v2", False):
        v2_params = {
            "steepness_fire": cfg.steepness_fire,
            "steepness_gj": cfg.steepness_gj,
            "steepness_input": cfg.steepness_input,
            "input_thresh": cfg.input_thresh,
            "leaky_slope": cfg.leaky_slope,
        }

    actor = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_action,
        internal_steps=cfg.twc_internal_steps,
        initial_thresholds=cfg.twc_trhesholds,
        initial_decays=cfg.twc_decays,
        rnd_init=cfg.rnd_init,
        use_V2=cfg.use_v2,
        log_stats=False,
        **v2_params,
    )

    # --- Críticos ---
    critic_1 = Critic(state_dim, action_dim, size=cfg.critic_hidden_layers)
    critic_2 = Critic(state_dim, action_dim, size=cfg.critic_hidden_layers)

    # --- Optimizadores ---
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(
        itertools.chain(critic_1.parameters(), critic_2.parameters()),
        lr=cfg.critic_lr,
    )

    # --- Engine TD3 ---
    engine = TD3Engine(
        gamma=cfg.gamma,
        tau=cfg.tau,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        policy_delay=cfg.policy_delay,
        target_policy_noise=cfg.target_noise,
        target_noise_clip=cfg.noise_clip,
        device=cfg.device,
        use_seq_optimized=True
    )

    # --- Replay buffer secuencial con PER ---
    replay_buf = SequenceBuffer(capacity=cfg.replay_buffer_size)

    # Prefill buffer with 5 random episodes for quicker BPTT sampling during profiling
    env_step = env.step
    env_reset = env.reset
    sample_action = env.action_space.sample
    for ep in range(5):
        obs, _ = env_reset(seed=seed + ep)
        done = False
        while not done:
            action = sample_action()
            next_obs, reward, terminated, truncated, _ = env_step(action)
            replay_buf.store(obs, action, reward, next_obs, terminated, truncated)
            obs = next_obs
            done = terminated or truncated

    # --- Ruido OU ---
    noise = OUNoise(
        size=env.action_space.shape,
        mu=0.0,
        theta=0.15,
        sigma_init=cfg.ou_sigma_init,
        sigma_min=cfg.ou_sigma_end,
        decay_steps=int(cfg.max_train_steps * 0.7),
        dt=1.0,
        seed=cfg.seed,
    )

    # --- Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"twc_mcc_V{cfg.use_v2}_Profiling_{timestamp}"
    log_dir = f"out/runs/td3_Profiling/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    config_path = os.path.join(log_dir, "full_config.json")
    with open(config_path, "w") as f:
        f.write(cfg.to_json())

    return env, replay_buf, engine, noise, writer, log_dir, timestamp


def run_training_with_profiling(
    cfg: TD3Config,
    use_torch_profiler: bool,
    use_cpu_profiler: bool,
):
    """
    Envuelve la llamada a td3_train con:
      - cProfile (CPU)
      - torch.profiler (CPU+CUDA)
    según los flags.
    """
    env, replay_buf, engine, noise, writer, log_dir, timestamp = build_training_objects(cfg)

    # --- cProfile ---
    cpu_prof = None
    if use_cpu_profiler:
        cpu_prof = cProfile.Profile()
        cpu_prof.enable()

    # --- torch.profiler ---
    if use_torch_profiler:
        prof_dir = os.path.join(log_dir, "torch_profiler")
        os.makedirs(prof_dir, exist_ok=True)

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        # ATENCIÓN: usar una config con pocos pasos (max_train_steps bajo)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,   # para que sea más liviano
            with_stack=False,       # idem
            on_trace_ready=tensorboard_trace_handler(prof_dir)
        ) as torch_prof:
            td3_train(
                env=env,
                replay_buf=replay_buf,
                engine=engine,
                writer=writer,
                timestamp=timestamp,
                config=cfg,
                OUNoise=noise,
            )

    else:
        # Entrenamiento normal (sin torch.profiler)
        td3_train(
            env=env,
            replay_buf=replay_buf,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            OUNoise=noise,
            use_PER=False,
        )

    # --- Post-procesar cProfile ---
    if use_cpu_profiler and cpu_prof is not None:
        cpu_prof.disable()
        stats = pstats.Stats(cpu_prof).sort_stats("tottime")

        cpu_profile_path = os.path.join(log_dir, "cpu_profile.txt")
        with open(cpu_profile_path, "w") as f:
            stats.stream = f
            stats.print_stats(80)  # top 80 funciones más costosas

        print(f"[cProfile] CPU profile guardado en: {cpu_profile_path}")


def main():
    args = parse_args()
    config_path = Path(args.config_path)
    print(f"[INFO] Using config: {config_path}")

    cfg = TD3Config()
    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)
        cfg = cfg.load(config_data)

    # Overwrite de pasos máximos solo para profiling corto
    if args.profile_max_train_steps is not None:
        print(f"[INFO] Overriding cfg.max_train_steps -> {args.profile_max_train_steps} (profiling run)")
        cfg.max_train_steps = args.profile_max_train_steps

    print(cfg)
    run_training_with_profiling(
        cfg,
        use_torch_profiler=args.torch_profiler,
        use_cpu_profiler=args.cpu_profiler,
    )


if __name__ == "__main__":
    main()
