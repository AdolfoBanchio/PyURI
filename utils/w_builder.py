import json
from typing import Dict, Tuple, List
import torch
import math
import warnings

def build_tw_matrices(
    data: Dict,
    *,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    init: str = "kaiming_uniform",   # "xavier_uniform", "normal", "constant"
    gain: float = 1.0,
    const_value: float = 1.0,
    random_seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Build channel-separated weight matrices for a 3-layer TWC split:
      - Input layer:   data['groups']['input']
      - Hidden layer:  data['groups']['hidden']
      - Output layer:  data['groups']['output']
    Channels: data['channels'] must map {"EX":0,"IN":1,"GJ":2}.

    Returns a dict with:
      Masks (bool):  M_in2hid_EX/IN/GJ, M_hid_EX/IN/GJ, M_hid2out_EX/IN/GJ
      Weights (float): W_in2hid_EX/IN/GJ, W_hid_EX/IN/GJ, W_hid2out_EX/IN/GJ

    Notes
    -----
    - We assume the **source activity** for all neurons is read from **channel 0** (the FIURI output).
    - Edges are routed into the target **channel** indicated by type (EX→0, IN→1, GJ→2).
    - GJ are kept **one-way** (no automatic symmetry), as you requested.
    - Any edge that doesn’t belong to (input→hidden | hidden→hidden | hidden→output) is ignored with a warning.
    """

    torch.manual_seed(random_seed)

    name2idx: Dict[str, int] = data["neurons"]
    groups: Dict[str, List[str]] = data["groups"]
    chan: Dict[str, int] = data["channels"]
    edges: List[Dict[str, str]] = data["edges"]

    # --- Validate channels map
    for key in ("EX", "IN", "GJ"):
        if key not in chan:
            raise ValueError(f"channels map must include '{key}'")

    # --- Helper: local index within a group
    def local_index(name: str, group_list: List[str]) -> int:
        try:
            return group_list.index(name)
        except ValueError:
            return -1

    IN_NEUR = groups["input"]
    HID_NEUR = groups["hidden"]
    OUT_NEUR = groups["output"]

    N_in, N_hid, N_out = len(IN_NEUR), len(HID_NEUR), len(OUT_NEUR)

    # Flattened sizes with 3 channels per cell
    def flat_size(n_cells: int) -> int:
        return n_cells * 3

    pre_in_n   = flat_size(N_in)
    pre_hid_n  = flat_size(N_hid)
    pre_out_n  = flat_size(N_out)  # rarely used as source here

    post_in_n  = pre_in_n
    post_hid_n = pre_hid_n
    post_out_n = pre_out_n

    # Create masks (bool) for each block and channel type
    def zeros_mask(pre_n, post_n):
        return torch.zeros((pre_n, post_n), dtype=torch.bool, device=device)

    M_in2hid_EX = zeros_mask(pre_in_n,  post_hid_n)
    M_in2hid_IN = zeros_mask(pre_in_n,  post_hid_n)
    M_in2hid_GJ = zeros_mask(pre_in_n,  post_hid_n)

    M_hid_EX    = zeros_mask(pre_hid_n, post_hid_n)
    M_hid_IN    = zeros_mask(pre_hid_n, post_hid_n)
    M_hid_GJ    = zeros_mask(pre_hid_n, post_hid_n)

    M_hid2out_EX = zeros_mask(pre_hid_n, post_out_n)
    M_hid2out_IN = zeros_mask(pre_hid_n, post_out_n)
    M_hid2out_GJ = zeros_mask(pre_hid_n, post_out_n)

    # post index helper for a (cell_id, channel_id)
    def post_idx(cell_id: int, channel_id: int) -> int:
        return cell_id * 3 + channel_id

    # When reading from a source neuron, we read its output from channel 0
    SRC_CH = 0

    # Fill masks from edges
    for e in edges:
        src = e["src"]; dst = e["dst"]; etype = e["type"]  # "EX" | "IN" | "GJ"
        if etype not in ("EX", "IN", "GJ"):
            warnings.warn(f"Unknown edge type '{etype}' for {src}->{dst}; skipping.")
            continue

        # Figure out which block this edge belongs to
        src_g = ("input"  if src in IN_NEUR  else
                 "hidden" if src in HID_NEUR else
                 "output" if src in OUT_NEUR else None)
        dst_g = ("input"  if dst in IN_NEUR  else
                 "hidden" if dst in HID_NEUR else
                 "output" if dst in OUT_NEUR else None)

        if src_g is None or dst_g is None:
            warnings.warn(f"Edge {src}->{dst} references neuron not present in groups; skipping.")
            continue

        # Map to local IDs in their respective groups
        if src_g == "input":
            src_local = local_index(src, IN_NEUR)
            pre_base  = post_idx(src_local, SRC_CH)  # within input layer (flattened)
        elif src_g == "hidden":
            src_local = local_index(src, HID_NEUR)
            pre_base  = post_idx(src_local, SRC_CH)  # within hidden layer
        else:
            src_local = local_index(src, OUT_NEUR)
            pre_base  = post_idx(src_local, SRC_CH)  # within output layer (rare as source)

        if dst_g == "input":
            dst_local = local_index(dst, IN_NEUR)
            post_base = post_idx(dst_local, chan[etype])
        elif dst_g == "hidden":
            dst_local = local_index(dst, HID_NEUR)
            post_base = post_idx(dst_local, chan[etype])
        else:
            dst_local = local_index(dst, OUT_NEUR)
            post_base = post_idx(dst_local, chan[etype])

        # Place into the right block mask
        if src_g == "input" and dst_g == "hidden":
            tgt = {"EX": M_in2hid_EX, "IN": M_in2hid_IN, "GJ": M_in2hid_GJ}[etype]
            tgt[pre_base, post_base] = True

        elif src_g == "hidden" and dst_g == "hidden":
            tgt = {"EX": M_hid_EX, "IN": M_hid_IN, "GJ": M_hid_GJ}[etype]
            tgt[pre_base, post_base] = True

        elif src_g == "hidden" and dst_g == "output":
            tgt = {"EX": M_hid2out_EX, "IN": M_hid2out_IN, "GJ": M_hid2out_GJ}[etype]
            tgt[pre_base, post_base] = True

        else:
            # All other cross-group edges are ignored for this 3-layer split
            warnings.warn(f"Ignoring edge {src_g}->{dst_g}: {src}->{dst}")

    # --- Initialize weights wherever mask is True
    def init_from_mask(mask: torch.Tensor) -> torch.Tensor:
        W = torch.zeros(mask.shape, dtype=dtype, device=device)
        nnz = mask.sum().item()
        if nnz == 0:
            return W

        # Select init distribution
        if init == "kaiming_uniform":
            # Fan-in for post; we approximate with mask-based fan_in per column
            # Simpler: use a single bound based on fan_in ≈ max(1, mask.sum(0).max())
            fan_in = int(mask.sum(dim=0).max().clamp(min=1).item())
            bound = gain * math.sqrt(6.0 / fan_in)
            W[mask] = (torch.rand(nnz, device=device, dtype=dtype) * 2 - 1) * bound

        elif init == "xavier_uniform":
            fan_in = int(mask.sum(dim=0).max().clamp(min=1).item())
            fan_out = int(mask.sum(dim=1).max().clamp(min=1).item())
            bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
            W[mask] = (torch.rand(nnz, device=device, dtype=dtype) * 2 - 1) * bound

        elif init == "normal":
            W[mask] = torch.randn(nnz, device=device, dtype=dtype) * gain

        elif init == "constant":
            W[mask] = const_value

        else:
            raise ValueError(f"Unknown init '{init}'")
        return W

    out = {
        # Masks
        "M_in2hid_EX": M_in2hid_EX, "M_in2hid_IN": M_in2hid_IN, "M_in2hid_GJ": M_in2hid_GJ,
        "M_hid_EX":    M_hid_EX,    "M_hid_IN":    M_hid_IN,    "M_hid_GJ":    M_hid_GJ,
        "M_hid2out_EX": M_hid2out_EX, "M_hid2out_IN": M_hid2out_IN, "M_hid2out_GJ": M_hid2out_GJ,
    }

    # Weights (random where mask=True, 0 elsewhere)
    # Build weights and an iterable spec including source/target for each matrix
    specs = [
        ("input",  "hidden", "EX", "W_in2hid_EX", M_in2hid_EX),
        ("input",  "hidden", "IN", "W_in2hid_IN", M_in2hid_IN),
        ("input",  "hidden", "GJ", "W_in2hid_GJ", M_in2hid_GJ),

        ("hidden", "hidden", "EX", "W_hid_EX",    M_hid_EX),
        ("hidden", "hidden", "IN", "W_hid_IN",    M_hid_IN),
        ("hidden", "hidden", "GJ", "W_hid_GJ",    M_hid_GJ),

        ("hidden", "output", "EX", "W_hid2out_EX", M_hid2out_EX),
        ("hidden", "output", "IN", "W_hid2out_IN", M_hid2out_IN),
        ("hidden", "output", "GJ", "W_hid2out_GJ", M_hid2out_GJ),
    ]

    connections = []
    for src, dst, etype, w_name, mask in specs:
        W = init_from_mask(mask)
        out[w_name] = W
        # Provide an iterable description to easily attach connections later
        connections.append({
            "name": w_name,
            "type": etype,
            "source": src,
            "target": dst,
            "weight": W,
            "mask": mask,
        })

    out["connections"] = connections

    return out
