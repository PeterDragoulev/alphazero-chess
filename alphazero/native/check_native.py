"""
Check native MCTS (fastmcts) against the Python reference (mcts.PyMCTS).

The reference gets a board whose legal moves come out in the native order
(from, to, promotion), so both trees see identical inputs; with the same
network, visit counts should then match up to float rounding.

    .venv/bin/python native/check_native.py
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import torch

from mcts import NativeMCTS, PyMCTS
from network import Evaluator, load_checkpoint


class SortedBoard(chess.Board):
    @property
    def legal_moves(self):
        return sorted(super().legal_moves,
                      key=lambda m: (m.from_square, m.to_square, m.promotion or 0))


def run(tree, ev, sims, batch):
    done = 0
    while done < sims:
        if batch:
            p, n = tree.select_leaves(min(batch, sims - done))
            done += n
            if p:
                lg, v = ev(np.stack(p))
                tree.expand_leaves(lg, v)
        else:
            p = tree.select_leaf()
            done += 1
            if p is not None:
                lg, v = ev(p[None])
                tree.expand_backup(lg[0], float(v[0]))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from config import CFG
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default = os.path.join(here, CFG.checkpoint)
    if not os.path.exists(default):
        default = os.path.join(here, CFG.pretrained_weights)
    model, _ = load_checkpoint(sys.argv[1] if len(sys.argv) > 1 else default, dev)
    ev = Evaluator(model, dev)
    rng = random.Random(3)
    games = []
    for _ in range(12):                      # random game prefixes, incl. repetitions
        b = SortedBoard()
        for _ in range(rng.randint(0, 60)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(rng.choice(ms))
        if not b.is_game_over():
            games.append(b)
    games.append(SortedBoard("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"))        # endgame
    games.append(SortedBoard("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"))  # mate in 1

    identical = total = 0
    worst = 0.0
    for batch in (0, 16):
        for b in games:
            py_t = PyMCTS(b.copy())
            nat_t = NativeMCTS(chess.Board(b.fen()) if not b.move_stack else _plain(b))
            for step in range(3):            # search, play the best move, reuse the tree
                run(py_t, ev, 400, batch)
                run(nat_t, ev, 400, batch)
                pm, pn = py_t.visit_counts()
                nm, nn = nat_t.visit_counts()
                assert [m.uci() for m in pm] == [m.uci() for m in nm], "move order differs"
                total += 1
                identical += int(np.array_equal(pn, nn))
                worst = max(worst, np.abs(pn - nn).sum() / max(pn.sum(), 1))
                mv = py_t.best_move(0.0)
                if py_t.board.is_game_over() or mv != nat_t.best_move(0.0):
                    break
                py_t.advance(mv)
                nat_t.advance(mv)
                if py_t.board.is_game_over():
                    break
    print(f"{identical}/{total} searches with identical root visit counts; "
          f"worst difference {worst:.1%} of visits")

    b = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 9")
    for cls in (PyMCTS, NativeMCTS):
        t = cls(b.copy())
        run(t, ev, 400, 16)
        t = cls(b.copy())
        t0 = time.perf_counter()
        run(t, ev, 3200, 16)
        dt = time.perf_counter() - t0
        print(f"{cls.__name__:10s} 3200 sims, batch 16: {dt:.2f}s = {3200 / dt:,.0f} sims/s")


def _plain(b):
    """Same game on a plain chess.Board (history kept, for repetition keys)."""
    p = chess.Board(b.root().fen())
    for m in b.move_stack:
        p.push(m)
    return p


if __name__ == "__main__":
    main()
