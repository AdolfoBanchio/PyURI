from utils import twc_builder
import torch

path = "twc_actor_50000.pth"

state_dict = torch.load(path)

for name, value in state_dict.items():
    print(f"{name}: {value}")
