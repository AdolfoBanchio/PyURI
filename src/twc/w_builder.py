import json
from typing import Dict, Tuple, List
import torch
import numpy as np

def _block_of(src: str, dst: str, groups: Dict[str, list]) -> str | None:
    if src in groups["input"]  and dst in groups["hidden"]: return "in2hid"
    if src in groups["hidden"] and dst in groups["hidden"]: return "hid"
    if src in groups["hidden"] and dst in groups["output"]: return "hid2out"
    return None  # ignore any other cross-group edge

def build_tw_matrices(spec: Dict) -> Dict[str, torch.Tensor]:
    """
    Build EX/IN masks for the three blocks:
      - in2hid:  (n_in,  n_hid)
      - hid:     (n_hid, n_hid)
      - hid2out: (n_hid, n_out)

    Returns:
      masks: {
        "in2hid": {"EX": mask_ex, "IN": mask_in},
        "hid":    {"EX": mask_ex, "IN": mask_in},
        "hid2out":{"EX": mask_ex, "IN": mask_in},
      }
      sizes: {"n_in": int, "n_hid": int, "n_out": int}
    """
    neurons: Dict[str, int] = spec["neurons"]
    groups:  Dict[str, list] = spec["groups"]
    edges    = spec["edges"]

    n_in  = len(groups["input"])
    n_hid = len(groups["hidden"])
    n_out = len(groups["output"])

    # Build index maps that match the order in groups (so row/col align with your layer ordering)
    idx_in  = {name: i for i, name in enumerate(groups["input"])}
    idx_hid = {name: i for i, name in enumerate(groups["hidden"])}
    idx_out = {name: i for i, name in enumerate(groups["output"])}

    def empty_pair(shape: Tuple[int, int]):
        # float32 masks (0.0 / 1.0)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    # Initialize all masks
    in2hid_EX, in2hid_IN   = empty_pair((n_in,  n_hid))
    hid_EX,    hid_IN      = empty_pair((n_hid, n_hid))
    hid2out_EX, hid2out_IN = empty_pair((n_hid, n_out))

    # Fill masks from edges
    for e in edges:
        src = e["src"]
        dst = e["dst"]
        et  = e["type"]   # "EX", "IN", or "GJ"
        if et not in ("EX", "IN"):
            continue  # ignore GJ for mask building

        block = _block_of(src, dst, groups)
        if block is None:
            # Not one of the three blocks we care about (e.g., input->output direct)
            continue

        if block == "in2hid":
            i = idx_in[src]
            j = idx_hid[dst]
            if et == "EX":
                in2hid_EX[i, j] = 1.0
            else:
                in2hid_IN[i, j] = 1.0

        elif block == "hid":
            i = idx_hid[src]
            j = idx_hid[dst]
            if et == "EX":
                hid_EX[i, j] = 1.0
            else:
                hid_IN[i, j] = 1.0

        elif block == "hid2out":
            i = idx_hid[src]
            j = idx_out[dst]
            if et == "EX":
                hid2out_EX[i, j] = 1.0
            else:
                hid2out_IN[i, j] = 1.0

    masks_np = {
        "in2hid":  {"EX": in2hid_EX,  "IN": in2hid_IN},
        "hid":     {"EX": hid_EX,     "IN": hid_IN},
        "hid2out": {"EX": hid2out_EX, "IN": hid2out_IN},
    }
    sizes = {"n_in": n_in, "n_hid": n_hid, "n_out": n_out}

    # Optional: return torch tensors instead (keeps same layout)
    import torch
    to_t = lambda a: torch.from_numpy(a)
    masks_t = {blk: {ch: to_t(arr) for ch, arr in d.items()} for blk, d in masks_np.items()}
    return masks_t, sizes


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import json
    path = "TWC_fiu.json"
    # Example: load from file or paste dict directly
    with open(path, "r") as f:
         spec = json.load(f)

    masks, sizes = build_tw_matrices(spec)
    print("sizes:", sizes)
    for blk in ("in2hid", "hid", "hid2out"):
        print(f"\n[{blk}]")
        print("EX mask:\n", masks[blk]["EX"])
        print("IN mask:\n", masks[blk]["IN"])