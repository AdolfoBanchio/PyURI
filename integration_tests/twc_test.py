import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt

from utils.twc_builder import build_twc
from utils.twc_io import mcc_obs_encoder, mcc_obs_encoder_speed_weighted, twc_out_2_mcc_action, POS_MIN, POS_MAX, VEL_MAX, twc_out_2_mcc_action_tanh


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Build TWC for MCC with logging enabled
    twc = build_twc(
        obs_encoder=mcc_obs_encoder_speed_weighted,
        action_decoder=twc_out_2_mcc_action_tanh,
        internal_steps=3,
        log_stats=True,
    )
    twc.eval()

    # Generate a short synthetic observation sequence (position, velocity)
    # Cover both sides of valley and velocities in [-VEL_MAX, VEL_MAX]
    T = 100
    pos = torch.linspace(POS_MIN, POS_MAX, T)
    vel = (VEL_MAX * 0.95) * torch.sin(torch.linspace(0, 4 * torch.pi, T))

    actions = []
    with torch.no_grad():
        for t in range(T):
            obs_t = torch.tensor([[pos[t].item(), vel[t].item()]], dtype=torch.float32)
            a = twc(obs_t)  # logs internal states each call
            actions.append(a.squeeze().item())

    # Extract monitor logs into (T, N) tensors per layer
    monitor = twc.monitor
    layers = ['in', 'hid', 'out']
    series = {}
    for L in layers:
        in_states = torch.stack([step['in_state'][0] for step in monitor[L]], dim=0)  # (T, N)
        out_states = torch.stack([step['out_state'][0] for step in monitor[L]], dim=0)  # (T, N)
        series[L] = (in_states, out_states)

    # Plot per-layer time series for in/out state of each neuron
    for L in layers:
        in_states, out_states = series[L]
        N = in_states.shape[1]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for i in range(N):
            ax1.plot(in_states[:, i].cpu().numpy(), label=f'E_{i}')
            ax2.plot(out_states[:, i].cpu().numpy(), label=f'O_{i}')
        ax1.set_title(f'{L.upper()} layer: Internal state (E) per neuron')
        ax2.set_title(f'{L.upper()} layer: Output state (O) per neuron')
        ax2.set_xlabel('Time step')
        ax1.set_ylabel('E')
        ax2.set_ylabel('O')
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        # Keep legends compact
        ax1.legend(ncol=max(1, N // 4), fontsize='small')
        ax2.legend(ncol=max(1, N // 4), fontsize='small')
        plt.tight_layout()
        save_path = os.path.join(out_dir, f'twc_{L}_states.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f'Saved {save_path}')

    # Save action trace for reference
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(actions, label='action(t)')
    ax.set_title('Decoded action over time')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Torque')
    ax.grid(True, alpha=0.3)
    ax.legend()
    act_path = os.path.join(out_dir, 'twc_actions.png')
    plt.tight_layout()
    plt.savefig(act_path)
    plt.close(fig)
    print(f'Saved {act_path}')


if __name__ == '__main__':
    main()
