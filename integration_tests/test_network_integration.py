import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Optional, Iterable, Union
from utils.twc_builder import build_TWC
from bindsnet.analysis.visualization import summary

net = build_TWC()


print(summary(net=net))

print("\nNetwork Parameters:\n")
for name, param in net.named_parameters():
    print(f"Name: {name}")
    print(f"  Shape: {param.shape}")
    print(f"  Dtype: {param.dtype}")
    print(f"  Device: {param.device}")
    print(f"  Requires grad: {param.requires_grad}")
    print(f"  Data (first 5 elements): {param.flatten()[:5].tolist()}")
    print("-" * 40)
