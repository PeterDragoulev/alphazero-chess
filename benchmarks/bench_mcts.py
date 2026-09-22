"""
bench_mcts.py — reproduces the MCTS self-play speedup numbers.

Runs the same self-play workload (64 games in lockstep, 128 simulations per
move, the pretrained net, Dirichlet noise at the root) three ways:

  1. unbatched + copy    one network call per leaf; the search copies the
                         board for every simulation (the naive starting point)
  2. batched   + copy    one GPU call per simulation round, evaluating the
                         leaves of all 64 games together
  3. batched   + push/pop  (current alphazero/mcts.py) search board is pushed
                         down the path and popped back, legal moves generated
                         once per leaf

mcts_copy_baseline.py is the tree before the push/pop change; the search
results are identical (same visit counts), only the speed differs.

    python bench_mcts.py                   # all three, ~2-3 min on a 3070 Ti
    python bench_mcts.py --moves 2         # quicker, noisier
"""

import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(HERE, "..", "alphazero")
sys.path.insert(0, ENGINE_DIR)

import chess                                            # noqa: E402
import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

from config import CFG                                  # noqa: E402
from network import Evaluator, load_checkpoint          # noqa: E402


def load_mcts(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MCTS


def run(MCTS, ev, batched, games, sims, moves, seed=0):
    """Returns simulations per second."""
    np.random.seed(seed)
    trees = [MCTS(chess.Board(), add_noise=True) for _ in range(games)]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(moves):
        for _ in range(sims):
            if batched:
                pending, planes = [], []
                for t in trees:
                    p = t.select_leaf()
                    if p is not None:
                        pending.append(t)
                        planes.append(p)
                if pending:
                    logits, values = ev(np.stack(planes))
                    for i, t in enumerate(pending):
                        t.expand_backup(logits[i], float(values[i]))
            else:
                for t in trees:
                    p = t.select_leaf()
                    if p is not None:
                        logits, values = ev(p[None])
                        t.expand_backup(logits[0], float(values[0]))
        for t in trees:
            t.advance(t.best_move(temperature=1.0))
    return games * sims * moves / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=os.path.join(ENGINE_DIR, CFG.pretrained_weights))
    ap.add_argument("--games", type=int, default=CFG.parallel_games)
    ap.add_argument("--sims", type=int, default=CFG.simulations)
    ap.add_argument("--moves", type=int, default=6,
                    help="move cycles timed for the batched runs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    ev = Evaluator(model, device)
    for b in (1, args.games):                           # warm up CUDA kernels
        ev(np.zeros((b, 19, 8, 8), np.uint8))

    old = load_mcts(os.path.join(HERE, "mcts_copy_baseline.py"))
    new = load_mcts(os.path.join(ENGINE_DIR, "mcts.py"))
    print(f"device {device} | {args.games} games x {args.sims} sims/move\n")

    # The unbatched run is ~17x slower, so time a single move cycle.
    rows = [("unbatched + copy", run(old, ev, False, args.games, args.sims, 1))]
    rows.append(("batched   + copy",
                 run(old, ev, True, args.games, args.sims, args.moves)))
    rows.append(("batched   + push/pop",
                 run(new, ev, True, args.games, args.sims, args.moves)))

    base = rows[0][1]
    print(f"{'variant':22s} {'sims/s':>8s} {'vs naive':>9s}")
    for name, rate in rows:
        print(f"{name:22s} {rate:8.0f} {rate / base:8.1f}x")
    print(f"\npush/pop over batched+copy: {rows[2][1] / rows[1][1] - 1:+.0%}")


if __name__ == "__main__":
    main()
