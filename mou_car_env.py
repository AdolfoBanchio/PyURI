import gymnasium as gym
from utils.twc_builder import build_twc
from utils.twc_io_wrapper import TwcIOWrapper, mountaincar_pair_encoder
from FIURI_node import FIURI_node
import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = build_twc()

mc_wrapper = TwcIOWrapper(
    net=net,
    device=dev,
    obs_encoder=mountaincar_pair_encoder(),
)
print(mc_wrapper.net)

env = gym.make("MountainCarContinuous-v0", render_mode="human")  # default goal_velocity=0
obs, info = env.reset(seed=123)

print(f"starting obs {obs}")

episode_over = False
total_reward = 0
episode = 0

while not episode_over:
    print(f"== episode {episode} ==")

    out_s = mc_wrapper.step(obs)
    action =  mc_wrapper.decode_action(out_s)

    obs, reward, terminated, truncated, info = env.step(action.detach().numpy())
    print(f"obs from env {obs}")
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    total_reward += reward
    episode_over = terminated or truncated
    episode +=1

print(f"episode over, total reward: {total_reward}")

