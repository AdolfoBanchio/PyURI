import math, warnings, torch
from typing import Dict, List, Tuple


def build_tw_edges(
    data: Dict,
    *,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    init: str = "kaiming_uniform",   # "xavier_uniform", "normal", "constant"
    gain: float = 1.0,
    const_value: float = 1.0,
    random_seed: int = 0,
    sort_by_dst: bool = True,
) -> Dict:
    """
    Build channel-separated SPARSE edge lists for a 3-block split:
      - input -> hidden
      - hidden -> hidden
      - hidden -> output
    Channels: data['channels'] must map {"EX":0,"IN":1,"GJ":2} (only used for validation).

    Returns:
      {
        "sizes": {"n_in":N_in, "n_hid":N_hid, "n_out":N_out},
        "in2hid": {"EX":{"src":(E,), "dst":(E,), "w":(E,)}, "IN":{...}, "GJ":{...}},
        "hid":    {"EX":{...}, "IN":{...}, "GJ":{...}},
        "hid2out":{"EX":{...}, "IN":{...}, "GJ":{...}}
      }
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

    IN_NEUR = groups["input"]
    HID_NEUR = groups["hidden"]
    OUT_NEUR = groups["output"]

    N_in, N_hid, N_out = len(IN_NEUR), len(HID_NEUR), len(OUT_NEUR)

    def local_index(name: str, group_list: List[str]) -> int:
        try:
            return group_list.index(name)
        except ValueError:
            return -1

    # Buckets: per block x channel we collect (src_list, dst_list)
    buckets = {
        "in2hid": {"EX": ([], []), "IN": ([], []), "GJ": ([], [])},
        "hid":    {"EX": ([], []), "IN": ([], []), "GJ": ([], [])},
        "hid2out":{"EX": ([], []), "IN": ([], []), "GJ": ([], [])},
    }

    # Classify edges into buckets
    for e in edges:
        src = e["src"]; dst = e["dst"]; etype = e["type"]  # "EX"|"IN"|"GJ"
        if etype not in ("EX", "IN", "GJ"):
            warnings.warn(f"Unknown edge type '{etype}' for {src}->{dst}; skipping.")
            continue

        # Which group is src/dst?
        src_g = ("input"  if src in IN_NEUR  else
                 "hidden" if src in HID_NEUR else
                 "output" if src in OUT_NEUR else None)
        dst_g = ("input"  if dst in IN_NEUR  else
                 "hidden" if dst in HID_NEUR else
                 "output" if dst in OUT_NEUR else None)

        if src_g is None or dst_g is None:
            warnings.warn(f"Edge {src}->{dst} references unknown neuron; skipping.")
            continue

        # Pick bucket
        if src_g == "input" and dst_g == "hidden":
            bucket = "in2hid"
            src_local = local_index(src, IN_NEUR)
            dst_local = local_index(dst, HID_NEUR)
        elif src_g == "hidden" and dst_g == "hidden":
            bucket = "hid"
            src_local = local_index(src, HID_NEUR)
            dst_local = local_index(dst, HID_NEUR)
        elif src_g == "hidden" and dst_g == "output":
            bucket = "hid2out"
            src_local = local_index(src, HID_NEUR)
            dst_local = local_index(dst, OUT_NEUR)
        else:
            # Ignore input->output or output->* for this 3-block split
            warnings.warn(f"Ignoring edge {src_g}->{dst_g}: {src}->{dst}")
            continue

        if src_local < 0 or dst_local < 0:
            warnings.warn(f"Edge {src}->{dst} could not be localized; skipping.")
            continue

        buckets[bucket][etype][0].append(src_local)  # src list
        buckets[bucket][etype][1].append(dst_local)  # dst list

    # Helper: init per-edge weights with fan-in/out aware bounds (per edge)
    def init_edge_weights(src: torch.LongTensor, dst: torch.LongTensor, n_pre: int, n_post: int) -> torch.Tensor:
        E = src.numel()
        if E == 0:
            return torch.zeros(0, device=device, dtype=dtype)

        # fan_in per post, fan_out per pre
        fan_in_per_post = torch.zeros(n_post, dtype=torch.long)
        fan_out_per_pre = torch.zeros(n_pre, dtype=torch.long)
        # Count degrees
        fan_in_per_post.index_add_(0, dst.cpu(), torch.ones(E, dtype=torch.long))
        fan_out_per_pre.index_add_(0, src.cpu(), torch.ones(E, dtype=torch.long))
        fan_in = fan_in_per_post[dst.cpu()].clamp(min=1).to(torch.float32)   # (E,)
        fan_out = fan_out_per_pre[src.cpu()].clamp(min=1).to(torch.float32) # (E,)

        # Create weights
        w = torch.empty(E, device=device, dtype=dtype)

        if init == "kaiming_uniform":
            # bound = gain * sqrt(6 / fan_in)
            bound = (gain * (6.0 / fan_in).sqrt()).to(dtype)
            # sample in [-bound, +bound] per edge
            w.uniform_(-1.0, 1.0).mul_(bound)
        elif init == "xavier_uniform":
            # bound = gain * sqrt(6 / (fan_in + fan_out))
            denom = (fan_in + fan_out).clamp(min=1.0)
            bound = (gain * (6.0 / denom).sqrt()).to(dtype)
            w.uniform_(-1.0, 1.0).mul_(bound)
        elif init == "normal":
            # std = gain / sqrt(fan_in)
            std = (gain / fan_in.sqrt()).to(dtype)
            # draw N(0,1) then scale per-edge
            w.normal_(0.0, 1.0).mul_(std)
        elif init == "constant":
            w.fill_(const_value)
        else:
            raise ValueError(f"Unknown init '{init}'")

        return w

    # Materialize tensors (src,dst,w) for each bucket/channel
    out = {
        "sizes": {"n_in": N_in, "n_hid": N_hid, "n_out": N_out},
        "in2hid": {"EX": {}, "IN": {}, "GJ": {}},
        "hid":    {"EX": {}, "IN": {}, "GJ": {}},
        "hid2out":{"EX": {}, "IN": {}, "GJ": {}},
    }

    def finalize_bucket(block: str, n_pre: int, n_post: int):
        for etype in ("EX", "IN", "GJ"):
            src_list, dst_list = buckets[block][etype]
            if len(src_list) == 0:
                out[block][etype] = {
                    "src": torch.zeros(0, dtype=torch.long, device=device),
                    "dst": torch.zeros(0, dtype=torch.long, device=device),
                    "w":   torch.zeros(0, dtype=dtype, device=device),
                }
                continue

            src = torch.tensor(src_list, dtype=torch.long, device=device)
            dst = torch.tensor(dst_list, dtype=torch.long, device=device)

            # Optional: sort by dst for scatter locality
            if sort_by_dst:
                perm = torch.argsort(dst)
                src = src[perm]; dst = dst[perm]

            w = init_edge_weights(src, dst, n_pre=n_pre, n_post=n_post)
            out[block][etype] = {"src": src, "dst": dst, "w": w}

    finalize_bucket("in2hid", n_pre=N_in,  n_post=N_hid)
    finalize_bucket("hid",    n_pre=N_hid, n_post=N_hid)
    finalize_bucket("hid2out",n_pre=N_hid, n_post=N_out)

    return out
