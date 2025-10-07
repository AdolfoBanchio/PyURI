import gymnasium as gym
from utils.twc_builder import build_TWC
from utils.twc_io_wrapper import TwcIOWrapper, mountaincar_pair_encoder
from FIURI_node import FIURI_node
from bindsnet.analysis.visualization import summary
import torch
import torch.nn as nn


net = build_TWC()
mc_wrapper = TwcIOWrapper(
    net=net,
    obs_encoder=mountaincar_pair_encoder(),
)
print(summary(mc_wrapper.net))

env = gym.make("MountainCarContinuous-v0", render_mode="human")  # default goal_velocity=0
obs, info = env.reset(seed=123)

print(f"starting obs {obs}")

episode_over = False
total_reward = 0
episode = 0

while not episode_over:
    print(f"== episode {episode} ==")

    action, out_s = mc_wrapper.step(obs)
    
    obs, reward, terminated, truncated, info = env.step(action.detach().numpy())
    print(f"obs from env {obs}")
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    total_reward += reward
    episode_over = terminated or truncated
    episode +=1

print(f"episode over, total reward: {total_reward}")
    
# Print inner (in_state) and out (out_state) state for all TWC neurons
for layer_name, layer in mc_wrapper.net.layers.items():
    if isinstance(layer, FIURI_node):
        in_s = layer.in_state.unsqueeze(0)
        out_s = layer.out_state.unsqueeze(0)
        print(f"Layer {layer_name} in_state: {in_s}")
        print(f"Layer {layer_name} in_state grad: {in_s.grad}")
        print(f"Layer {layer_name} out_state: {out_s}")
        print(f"Layer {layer_name} out_state grad: {out_s.grad}")
