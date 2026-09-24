"""
train.py — the AlphaZero loop: self-play → train → checkpoint, forever.

Built around interruption: work happens in blocks of CFG.games_per_block
self-play games followed by a proportional number of gradient steps, and
every block ends with an atomic checkpoint + replay-buffer flush. First
Ctrl+C stops at the next move cycle (a few seconds) and saves; second
Ctrl+C kills immediately (you lose only the current block's unsaved games).

Usage:
    python train.py                  # run until stopped
    python train.py --minutes 480    # stop (gracefully) after ~8 hours
    python train.py --blocks 10      # stop after 10 blocks
    python train.py --workers 4      # self-play in 4 parallel processes

Resume is automatic: just run train.py again. State lives in data/
(checkpoint.pt + buffer shards) — delete the directory to start over.
"""

import argparse
import csv
import os
import signal
import time

import torch
import torch.nn.functional as F

from config import CFG
from network import Evaluator, PolicyValueNet, load_checkpoint
from replay_buffer import ReplayBuffer
from selfplay import SelfPlayPool
from selfplay_workers import SelfPlayWorkers

LOG_PATH = os.path.join(CFG.data_dir, "train_log.csv")


class GracefulStop:
    """First SIGINT/SIGTERM requests a clean stop; second one is immediate."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\nStop requested — saving after current move cycle "
              "(Ctrl+C again to abort without saving)...", flush=True)


# -- state ---------------------------------------------------------------------


def load_state(device):
    """Returns (model, optimizer, scaler, games_done, steps_done)."""
    if os.path.exists(CFG.checkpoint):
        model, ckpt = load_checkpoint(CFG.checkpoint, device)
    else:
        model, ckpt = PolicyValueNet().to(device), {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr,
                                  weight_decay=CFG.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    games, steps = ckpt.get("games", 0), ckpt.get("steps", 0)
    if ckpt:
        print(f"Resumed checkpoint: {games:,} games, {steps:,} train steps")
    else:
        n = sum(p.numel() for p in model.parameters())
        print(f"Fresh model: {n:,} parameters")
    return model, optimizer, scaler, games, steps


def save_state(model, optimizer, scaler, games, steps):
    tmp = CFG.checkpoint + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "net": model.net_config,             # the model's own shape, not CFG's
        "games": games,
        "steps": steps,
    }, tmp)
    os.replace(tmp, CFG.checkpoint)


def log_block(games, steps, buffer_size, p_loss, v_loss, gpm):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "games", "steps", "buffer",
                        "policy_loss", "value_loss", "games_per_min"])
        w.writerow([int(time.time()), games, steps, buffer_size,
                    f"{p_loss:.4f}", f"{v_loss:.4f}", f"{gpm:.2f}"])


# -- training ------------------------------------------------------------------


def train_steps(model, optimizer, scaler, buffer, n_steps, device):
    """Run n_steps gradient updates; returns mean (policy_loss, value_loss)."""
    model.train()
    p_sum = v_sum = 0.0
    for _ in range(n_steps):
        planes, pi, z = buffer.sample(CFG.batch_size)
        x = torch.from_numpy(planes).to(device, non_blocking=True)
        pi_t = torch.from_numpy(pi).to(device, non_blocking=True)
        z_t = torch.from_numpy(z).to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits, values = model(x)
            policy_loss = -(pi_t * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
            value_loss = F.mse_loss(values.float(), z_t)
            loss = policy_loss + value_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        p_sum += policy_loss.item()
        v_sum += value_loss.item()
    return p_sum / max(n_steps, 1), v_sum / max(n_steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=None,
                        help="stop gracefully after this many minutes")
    parser.add_argument("--blocks", type=int, default=None,
                        help="stop after this many blocks")
    parser.add_argument("--workers", type=int, default=0,
                        help="self-play processes (0 = self-play in this process)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU — self-play will be very slow")

    os.makedirs(CFG.data_dir, exist_ok=True)
    stop = GracefulStop()
    model, optimizer, scaler, games, steps = load_state(device)
    buffer = ReplayBuffer()
    workers = pool = None
    if args.workers > 0:
        if not os.path.exists(CFG.checkpoint):
            save_state(model, optimizer, scaler, games, steps)   # workers load it
        workers = SelfPlayWorkers(args.workers, CFG.checkpoint, int(time.time()))
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        print(f"Self-play in {args.workers} worker processes")
    else:
        pool = SelfPlayPool(Evaluator(model, device))
        results = pool.results

    try:
        _train_loop(args, stop, model, optimizer, scaler, buffer, pool, workers,
                    results, games, steps, device)
    finally:
        if workers is not None:
            workers.close()
    print("Saved. Run train.py again to continue.")


def _train_loop(args, stop, model, optimizer, scaler, buffer, pool, workers,
                results, games, steps, device):
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    block = 0
    while not stop.requested:
        t0 = time.time()
        if workers is not None:
            positions = finished = 0
            for records, z in workers.get_games(CFG.games_per_block, stop):
                buffer.add_game(records, z)
                positions += len(records)
                finished += 1
                results["1-0" if z == 1 else "0-1" if z == -1 else "1/2-1/2"] += 1
        else:
            positions, finished = pool.play_block(buffer, CFG.games_per_block, stop)
        games += finished
        gpm = finished / max((time.time() - t0) / 60, 1e-9)

        p_loss = v_loss = float("nan")
        if buffer.size >= CFG.min_buffer and positions > 0:
            n_steps = max(1, round(positions * CFG.sample_ratio / CFG.batch_size))
            p_loss, v_loss = train_steps(model, optimizer, scaler,
                                         buffer, n_steps, device)
            steps += n_steps

        buffer.save()
        save_state(model, optimizer, scaler, games, steps)
        log_block(games, steps, buffer.size, p_loss, v_loss, gpm)
        r = results
        print(f"[block {block:4d}] games {games:,} ({gpm:.1f}/min) | "
              f"buffer {buffer.size:,} | steps {steps:,} | "
              f"loss p {p_loss:.3f} v {v_loss:.3f} | "
              f"W/D/L {r['1-0']}/{r['1/2-1/2']}/{r['0-1']}"
              + (f" | weight reloads {sorted(workers.reloads.values())}"
                 if workers is not None else ""), flush=True)

        block += 1
        if args.blocks is not None and block >= args.blocks:
            break
        if deadline is not None and time.time() >= deadline:
            print("Time limit reached.")
            break


if __name__ == "__main__":
    main()
