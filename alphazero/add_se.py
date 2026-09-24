"""
add_se.py — net surgery: add squeeze-and-excitation to every residual block
of a trained checkpoint, without changing what the net computes.

Each new SE module starts as an exact identity (network.SqueezeExcite), so
the converted net's outputs are bit-identical to the original's; training
then teaches the SE layers to use global context. The optimizer's Adam
moments are carried over for every existing weight (only the new SE weights
start fresh), so resuming training is smooth.

    .venv/bin/python add_se.py                       # data/checkpoint.pt in place
    .venv/bin/python add_se.py IN.pt OUT.pt

Back up the checkpoint first (the in-place form refuses to run without one
in models/ or backups/ matching it, unless --force).
"""

import argparse
import hashlib
import glob
import os

import numpy as np
import torch

from config import CFG
from network import PolicyValueNet


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=CFG.checkpoint)
    ap.add_argument("dst", nargs="?", default=None)
    ap.add_argument("--force", action="store_true", help="skip the backup check")
    args = ap.parse_args()
    dst = args.dst or args.src

    if dst == args.src and not args.force:
        mine = md5(args.src)
        backups = glob.glob("models/*.pt") + glob.glob("backups/*/checkpoint.pt")
        if not any(os.path.getsize(b) == os.path.getsize(args.src) and md5(b) == mine
                   for b in backups):
            raise SystemExit(f"No backup of {args.src} found in models/ or backups/; "
                             f"copy it first (or pass --force).")

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    net = ckpt.get("net", {})
    if net.get("se"):
        raise SystemExit("checkpoint already has SE blocks")
    channels, blocks = net.get("channels", CFG.channels), net.get("blocks", CFG.blocks)

    old = PolicyValueNet(channels, blocks, se=False)
    old.load_state_dict(ckpt["model"])
    new = PolicyValueNet(channels, blocks, se=True)
    missing, unexpected = new.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected, unexpected
    assert all(".se." in k for k in missing), missing

    # The converted net must compute exactly what the old one did.
    old.eval()
    new.eval()
    x = torch.rand(64, CFG.input_planes, 8, 8)
    with torch.no_grad():
        (p0, v0), (p1, v1) = old(x), new(x)
    assert torch.equal(p0, p1) and torch.equal(v0, v1), "SE is not an identity"

    # Carry Adam state over by parameter name (indices shift: SE params are
    # interleaved inside each block).
    old_names = [n for n, _ in old.named_parameters()]
    new_names = [n for n, _ in new.named_parameters()]
    if "optimizer" in ckpt:
        opt = ckpt["optimizer"]
        old_state = opt["state"]
        state = {}
        for j, name in enumerate(new_names):
            if name in old_names and old_names.index(name) in old_state:
                state[j] = old_state[old_names.index(name)]
        group = dict(opt["param_groups"][0])
        group["params"] = list(range(len(new_names)))
        ckpt["optimizer"] = {"state": state, "param_groups": [group]}
        # sanity: the rebuilt state loads into an optimizer over the new net
        probe = torch.optim.AdamW(new.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        probe.load_state_dict(ckpt["optimizer"])

    ckpt["model"] = new.state_dict()
    ckpt["net"] = new.net_config
    tmp = dst + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, dst)
    added = sum(p.numel() for n, p in new.named_parameters() if ".se." in n)
    total = sum(p.numel() for p in new.parameters())
    print(f"{dst}: SE added to {blocks} blocks (+{added:,} params, {total:,} total); "
          f"outputs identical on {len(x)} test positions; Adam state kept for "
          f"{len(old_names)} of {len(new_names)} parameter tensors")


if __name__ == "__main__":
    main()
