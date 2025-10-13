from utils import twc_builder
import torch

path = "twc_ppo_actor_final.pth"

state_dict = torch.load(path)

for name, value in state_dict.items():
    print(f"{name}: {value}")
