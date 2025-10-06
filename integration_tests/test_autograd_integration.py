import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FIURI_node import FIURI_node


def test_fiuri_autograd_single_layer_gradients():
    """Ensure autograd propagates to FIURI learnable parameters."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node = FIURI_node(
        num_cells=1,
        learn_threshold=True,
        learn_decay=True,
        sum_input=True,
        debug=False,
    ).to(device)

    node.set_batch_size(1)
    node.in_state = node.in_state.to(device)
    node.out_state = node.out_state.to(device)

    if node.threshold.grad is not None:
        node.threshold.grad.zero_()
    if node.decay.grad is not None:
        node.decay.grad.zero_()

    step1 = torch.tensor([[[2.0, 0.0, 0.0]]], device=device)
    node.forward(step1)

    step2 = torch.zeros_like(step1)
    node.forward(step2)

    loss = node.in_state.pow(2).mean() + node.out_state.pow(2).mean()
    loss.backward()

    threshold_grad = node.threshold.grad
    decay_grad = node.decay.grad

    assert threshold_grad is not None and torch.any(threshold_grad != 0)
    assert decay_grad is not None and torch.any(decay_grad != 0)

test_fiuri_autograd_single_layer_gradients()