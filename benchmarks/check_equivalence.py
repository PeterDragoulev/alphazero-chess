"""
check_equivalence.py — push/pop MCTS searches exactly like the copy baseline.

Runs both trees with the same seed and a small random-weight net (CPU, so it
is deterministic) over several plies, and asserts the root visit counts and
values match exactly — including positions with mates, stalemates, the
50-move rule and insufficient material. The current tree runs with FPU and
repetition draws switched off (fpu_reduction=None, repetition_draws=False):
those change the search on purpose; this check isolates the push/pop rewrite.

    python check_equivalence.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "alphazero"))

import chess                                            # noqa: E402
import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

from network import Evaluator, PolicyValueNet           # noqa: E402

FENS = {
    "start": chess.STARTING_FEN,
    "mate in 1": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "mate/stalemate": "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
    "50-move edge": "8/8/8/4k3/8/8/3K4/5R2 w - - 96 80",
}


CLASSIC = {"mcts_new": dict(fpu_reduction=None, repetition_draws=False)}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trace(mod, ev, fen, plies=6, sims=200):
    np.random.seed(1)
    board = chess.Board(fen)
    tree = mod.MCTS(board, add_noise=True, **CLASSIC.get(mod.__name__, {}))
    out = []
    for _ in range(plies):
        if board.outcome(claim_draw=True):
            break
        for _ in range(sims):
            p = tree.select_leaf()
            if p is not None:
                logits, values = ev(p[None])
                tree.expand_backup(logits[0], float(values[0]))
        out.append((tree.root.N.tolist(), tree.root.W.round(5).tolist()))
        tree.advance(tree.best_move(temperature=1.0))
    return out, board.fen()


def main():
    old = load(os.path.join(HERE, "mcts_copy_baseline.py"), "mcts_old")
    new = load(os.path.join(HERE, "..", "alphazero", "mcts.py"), "mcts_new")
    torch.manual_seed(0)
    ev = Evaluator(PolicyValueNet(channels=32, blocks=2), torch.device("cpu"))
    for name, fen in FENS.items():
        a, b = trace(old, ev, fen), trace(new, ev, fen)
        assert a == b, f"search diverged on {name}"
        print(f"identical: {name:16s} ({len(a[0])} plies searched)")
    print("OK — push/pop search matches the copy baseline exactly")


if __name__ == "__main__":
    main()
