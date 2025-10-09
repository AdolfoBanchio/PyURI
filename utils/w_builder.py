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
    use_json_weights: bool = False,
) -> Dict:
    """
    Build channel-separated SPARSE edge lists for a 3-block split:
      - input -> hidden
      - hidden -> hidden
      - hidden -> output
    Channels: data['channels'] must map {"EX":0,"IN":1,"GJ":2} (only used for validation).

    If `use_json_weights` is True, the weight for each edge is taken from
    the `weight` field in `data["edges"]`. Otherwise, weights are randomly
    initialized using the selected `init` scheme.

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
    # Optional: weights provided by JSON
    weights_buckets = {
        "in2hid": {"EX": [], "IN": [], "GJ": []},
        "hid":    {"EX": [], "IN": [], "GJ": []},
        "hid2out":{"EX": [], "IN": [], "GJ": []},
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
        if use_json_weights:
            try:
                wv = float(e.get("weight", const_value))
            except Exception:
                warnings.warn(f"Edge {src}->{dst} missing/invalid weight; defaulting to {const_value}.")
                wv = float(const_value)
            weights_buckets[bucket][etype].append(wv)

    # Helper: init per-edge weights with fan-in/out aware bounds (per edge)
    def init_edge_weights(src: torch.LongTensor, dst: torch.LongTensor, n_pre: int, n_post: int) -> torch.Tensor:
        E = src.numel()
        if E == 0:
            return torch.zeros(0, device=device, dtype=dtype)

        # Compute fan-in per post neuron and fan-out per pre neuron
        # Use bincount for simplicity and speed, keep work on CPU then index
        src_cpu = src.detach().cpu()
        dst_cpu = dst.detach().cpu()
        fan_in_per_post = torch.bincount(dst_cpu, minlength=n_post)
        fan_out_per_pre = torch.bincount(src_cpu, minlength=n_pre)
        fan_in = fan_in_per_post[dst_cpu].clamp_min(1).to(torch.float32)
        fan_out = fan_out_per_pre[src_cpu].clamp_min(1).to(torch.float32)

        if init == "constant":
            return torch.full((E,), const_value, device=device, dtype=dtype)

        if init == "kaiming_uniform":
            denom = fan_in  # per-edge
            bound = (gain * (6.0 / denom).sqrt()).to(device=device, dtype=dtype)
            return (torch.rand(E, device=device, dtype=dtype) * 2.0 - 1.0) * bound
        elif init == "xavier_uniform":
            denom = (fan_in + fan_out).clamp_min(1.0)
            bound = (gain * (6.0 / denom).sqrt()).to(device=device, dtype=dtype)
            return (torch.rand(E, device=device, dtype=dtype) * 2.0 - 1.0) * bound
        elif init == "normal":
            std = (gain / fan_in.sqrt()).to(device=device, dtype=dtype)
            return torch.randn(E, device=device, dtype=dtype) * std
        else:
            raise ValueError(f"Unknown init '{init}'")

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

            # Weights: from JSON or random init
            if use_json_weights:
                w_list = weights_buckets[block][etype]
                w = torch.tensor(w_list, dtype=dtype, device=device)
                if sort_by_dst:
                    w = w[perm]
            else:
                w = init_edge_weights(src, dst, n_pre=n_pre, n_post=n_post)
            out[block][etype] = {"src": src, "dst": dst, "w": w}

    finalize_bucket("in2hid", n_pre=N_in,  n_post=N_hid)
    finalize_bucket("hid",    n_pre=N_hid, n_post=N_hid)
    finalize_bucket("hid2out",n_pre=N_hid, n_post=N_out)

    return out
