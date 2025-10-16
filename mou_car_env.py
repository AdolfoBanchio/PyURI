import gymnasium as gym
from utils.twc_builder import build_twc
from utils.twc_io import mcc_obs_encoder, twc_out_2_mcc_action
from utils.MLP_models import Actor
import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
twc = build_twc(obs_encoder=mcc_obs_encoder,
                action_decoder=twc_out_2_mcc_action,
                log_stats=False)

actor = Actor(state_dim=2, action_dim=1)

path = "models/ddpg_actor_best.pth"
state_dict = torch.load(path)

actor.load_state_dict(state_dict=state_dict)

env = gym.make("MountainCarContinuous-v0", render_mode="human")  # default goal_velocity=0
obs, info = env.reset(seed=123)

for ep in range(1):
    obs, _ = env.reset()
    done, total_r = False, 0.0
    while not done:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1,2)
        a = actor(x)                       # (1,2)
        obs, r, terminated, truncated, _ = env.step(a.squeeze(0).detach().numpy())
        done = terminated or truncated
        total_r += r

