import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import json
import os
import itertools
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from td3 import TD3Engine
from utils import ReplayBuffer, OUNoise
from mlp import Critic
from twc import (
    build_twc,
    mcc_obs_encoder,
    twc_out_2_mcc_action,
)

# --- Environment ---
ENV = "MountainCarContinuous-v0"
SEED = 42
MAX_EPISODE        = 300
MAX_TIME_STEPS     = 999

# --- Hyperparameters ---
WARMUP_STEPS       = 10_000
BATCH_SIZE         = 128
NUM_UPDATE_LOOPS   = 2
UPDATE_EVERY       = 1

GAMMA              = 0.99
TAU                = 5e-3
ACTOR_LR           = 0.00028007729801810964
CRITIC_LR          = 0.004320799314236164

TWC_INTERNAL_STEPS = 1
CRITIC_HID_LAYERS  = [400, 300]
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA_START, SIGMA_END, SIGMA_DECAY_EPIS = 0.20, 0.05, 100


params = {
    'max_episode': MAX_EPISODE,
    'max_time_steps': MAX_TIME_STEPS,
    'warmup_steps': WARMUP_STEPS,
    'batch_size': BATCH_SIZE,
    'num_update_loops': NUM_UPDATE_LOOPS,
    'update_every': UPDATE_EVERY,
    'gamma': GAMMA,
    'tau': TAU,
    'actor_lr': ACTOR_LR,
    'critic_lr': CRITIC_LR,
    'critic_layers': CRITIC_HID_LAYERS,
    'target_policy_noise': 0.2,
    'target_noise_clip': 0.5,
    'env': ENV,
    'seed': SEED,
    'sigma_start': SIGMA_START,
    'sigma_end': SIGMA_END,
    'sigma_ep_end': SIGMA_DECAY_EPIS,
    'device': str(DEVICE),
}


def td3_train(env, replay_buf, ou_noise, engine, writer, timestamp):
    obs, _ = env.reset()
    ep_ret, ep_len = 0, 0
    best_ret = -np.inf
    total_steps = 0

    for e in tqdm(range(MAX_EPISODE)):
        obs, _ = env.reset()
        ou_noise.update_sigma(e)
        ou_noise.reset()    
        ep_reward = 0
        steps = 0
        engine.actor.reset()
        ep_actions = []
        for t in range(MAX_TIME_STEPS):
            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = engine.get_action(obs, action_noise=ou_noise.noise)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

            replay_buf.store(obs, action, reward, next_obs, terminated, truncated)
            obs = next_obs
            total_steps += 1

            if done:
                break
            
            if total_steps > WARMUP_STEPS and replay_buf.size >= BATCH_SIZE:
                ep_actions.append(action)
                for _ in range(NUM_UPDATE_LOOPS):
                    batch = replay_buf.sample(BATCH_SIZE, DEVICE)
                    actor_loss, critic_loss = engine.update_step(batch)
                    # Log losses
                    writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                    writer.add_scalar('Loss/Critic', critic_loss, total_steps)


        # Log episode return
        writer.add_scalar('Training/Episode_Return', ep_reward, total_steps)
        writer.add_scalar('Training/Episode_steps', steps, total_steps)
        writer.add_scalar('Training/AvgAction', np.mean(ep_actions), e)

        if (e+1) % 10 == 0:
            eval_ret, eval_avg_action = engine.evaluate_policy(env, episodes=10)
            # Log evaluation return
            writer.add_scalar('Evaluation/Return', eval_ret, total_steps)
            writer.add_scalar('Evaluation/AvgAction', eval_avg_action, total_steps)

            print(f"Evaluation after Episode {e+1}: {eval_ret:.2f}")
            if eval_ret > best_ret:
                best_ret = eval_ret
                torch.save(engine.actor.state_dict(), f"out/models/twc_td3_actor_best_{timestamp}.pth")
                print(f"New best evaluation reward: {best_ret:.2f}, models saved.")


def main():
    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = gym.make(ENV)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = build_twc(obs_encoder=mcc_obs_encoder,
                    action_decoder=twc_out_2_mcc_action,
                    internal_steps=TWC_INTERNAL_STEPS,
                    log_stats=False)

    critic_1 = Critic(state_dim, action_dim, size=CRITIC_HID_LAYERS)
    critic_2 = Critic(state_dim, action_dim, size=CRITIC_HID_LAYERS)


    buffer = ReplayBuffer(obs_dim=env.observation_space.shape[0],
                            act_dim=env.action_space.shape[0],
                            size=100_000,
                            keep=WARMUP_STEPS)

    ou_noise = OUNoise(action_dimension=env.action_space.shape[0],
                    mu=0,
                    theta=0.15,
                    sigma=SIGMA_START,
                    sigma_end=SIGMA_END,
                    sigma_decay_epis=SIGMA_DECAY_EPIS)

    actor_opt = torch.optim.Adam(actor.parameters(),  lr=ACTOR_LR)
    critic_opt = torch.optim.Adam(
        itertools.chain(critic_1.parameters(), critic_2.parameters()),
        lr=3e-4
    )

    td3 = TD3Engine(gamma=GAMMA,
                    tau=TAU,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    actor=actor,
                    critic_1=critic_1,
                    critic_2=critic_2,
                    actor_optimizer=actor_opt,
                    critic_optimizer=critic_opt,
                    policy_delay=UPDATE_EVERY,
                    target_policy_noise=0.21622134513436297	,
                    target_noise_clip=0.44117829502440176,
                    device=DEVICE)
    
    # Create a unique run directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"twc_td3_{timestamp}"
    writer = SummaryWriter(f'out/runs/td3/{run_name}')

    os.makedirs(writer.log_dir, exist_ok=True)
    params_path = os.path.join(writer.log_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    td3_train(env=env,
              replay_buf=buffer,
              ou_noise=ou_noise,
              engine=td3,
              writer=writer,
              timestamp=timestamp)
    
    print("Training completed.")

    env.close()
    # save final models
    torch.save(td3.actor.state_dict(), f"out/models/twc_td3_final_{timestamp}.pth")

    # Close tensorboard writer
    writer.close()

if __name__ == "__main__":
    main()