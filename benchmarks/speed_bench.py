"""
speed_bench.py — play-search speed (one tree, native MCTS) by virtual-loss
batch size, with the training-style evaluator and the frozen one (BatchNorm
folded, fp16, channels_last). Needs the native tree (alphazero/native/build.sh).

    python benchmarks/speed_bench.py                    # shipped weights, batches 4-32
    python benchmarks/speed_bench.py path/to.pt 16,32,64 frozen
"""
import os
import sys
import time

ENGINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alphazero")
sys.path.insert(0, ENGINE_DIR)
os.chdir(ENGINE_DIR)

import chess                                            # noqa: E402
import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

from config import CFG                                  # noqa: E402
from mcts import NativeMCTS                             # noqa: E402
from network import Evaluator, load_checkpoint          # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else CFG.pretrained_weights
BATCHES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8, 16, 32]
ONLY = sys.argv[3] if len(sys.argv) > 3 else None
dev = torch.device("cuda")
model, _ = load_checkpoint(path, dev)
model.eval()
evs = {"normal": Evaluator(model, dev), "frozen": Evaluator(model, dev, frozen=True)}
fens = ["r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 9",
        "r2q1rk1/1b2bppp/p2ppn2/1p6/3NP3/1BN1B3/PPP2PPP/R2Q1RK1 w - - 0 12",
        "8/5pk1/6p1/3R4/5P2/r5PK/8/8 w - - 0 45"]
for name, ev in evs.items():
    if ONLY and name != ONLY:
        continue
    for batch in BATCHES:
        rates = []
        for fen in fens:
            for rep in range(2):                        # first run warms up the graphs
                t = NativeMCTS(chess.Board(fen))
                done, t0 = 0, time.perf_counter()
                while done < 6400:
                    p, n = t.select_leaves(batch)
                    done += n
                    if p:
                        lg, v = ev(np.stack(p))
                        t.expand_leaves(lg, v)
                if rep:
                    rates.append(done / (time.perf_counter() - t0))
        print(f"{name:6s} batch {batch:2d}: {np.mean(rates):7,.0f} sims/s", flush=True)
