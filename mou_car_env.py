import gymnasium as gym
from utils.twc_builder import build_twc
from utils.twc_io import mcc_obs_encoder, twc_out_2_mcc_action
from utils.MLP_models import Actor
import torch
import math
from gymnasium.wrappers import RecordVideo
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
twc = build_twc(obs_encoder=mcc_obs_encoder,
                action_decoder=twc_out_2_mcc_action,
                log_stats=False)

actor = Actor(state_dim=2, action_dim=1, sizes=[400, 300])

path = "models/ddpg_actor_best.pth"
state_dict = torch.load(path)

actor.load_state_dict(state_dict=state_dict)

env = gym.make("MountainCarContinuous-v0")  # default goal_velocity=0
obs, info = env.reset(seed=123)

# make sure actor is on the right device
actor.to(dev)
actor.eval()

best_reward = -math.inf
best_seed = None
base_seed = 123
avg_rew = 0.0
# evaluate for 10 episodes
for ep in range(10):
    obs, _ = env.reset()
    done = False
    rew = 0.0
    with torch.no_grad():
        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)  # (1, state_dim)
            a = actor(x).squeeze(0).cpu().numpy()                                # (action_dim,)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            rew += r

    if rew > best_reward:
        best_reward = rew
    avg_rew += rew

avg_rew = avg_rew / 10

print(f"Evaluation endend, avg reward: {avg_rew}")

env.close()

print(f"Best reward: {best_reward} (seed={best_seed})")

# Re-run the best episode and save a video
record_folder = "videos"
record_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
record_env = RecordVideo(record_env, video_folder=record_folder, episode_trigger=lambda idx: True, name_prefix="best")

obs, _ = record_env.reset(seed=best_seed)
done = False
total_r = 0.0
with torch.no_grad():
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        a = actor(x).squeeze(0).cpu().numpy()
        obs, r, terminated, truncated, _ = record_env.step(a)
        done = terminated or truncated
        total_r += r

record_env.close()
print(f"Recorded best episode (reward={total_r}) to {record_folder}/")

