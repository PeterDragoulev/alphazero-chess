"""
elo_stockfish.py — estimate the engine's Elo against strength-limited Stockfish.

Stockfish's UCI_LimitStrength/UCI_Elo (1320-3190) gives an opponent on a
known scale. All games run in lockstep: our side's searches for every game
are batched into shared GPU calls (virtual loss within each tree), while each
game's Stockfish process thinks concurrently. Openings: 8 plies sampled from
our net's policy, all distinct, each played with both colors. Our engine runs
pure MCTS (no book, no tablebase) so the number reflects net + search.

    python elo_stockfish.py --elo 2700 --sf-time 3               # 10 games
    python elo_stockfish.py --elo 2900 --sims 6400 --sf-time 5

Needs a Stockfish binary (https://stockfishchess.org/download/): on PATH, in
$STOCKFISH, or passed with --stockfish.

Caveat: Stockfish's UCI_Elo is calibrated at longer time controls than a
fixed second per move, so treat the result as a rough rating, not a CCRL one.
"""

import argparse
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine
import numpy as np
import torch

from config import CFG
from encoding import encode_board, move_to_index
from mcts import MCTS
from network import Evaluator, load_checkpoint

STOCKFISH = os.environ.get("STOCKFISH") or shutil.which("stockfish") or "stockfish"
MAX_PLIES = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", type=int, required=True, help="Stockfish UCI_Elo (1320-3190)")
    ap.add_argument("--pairs", type=int, default=5, help="openings (each played twice)")
    ap.add_argument("--sims", type=int, default=1600, help="our MCTS simulations per move")
    ap.add_argument("--sf-time", type=float, default=1.0, help="Stockfish seconds per move")
    ap.add_argument("--checkpoint", default=(CFG.checkpoint if os.path.exists(CFG.checkpoint)
                                             else CFG.pretrained_weights))
    ap.add_argument("--stockfish", default=STOCKFISH, help="path to a Stockfish binary")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, dev)
    ev = Evaluator(model, dev)
    rng = random.Random(11)
    print(f"{args.checkpoint} ({ckpt.get('games', 0):,} games) at {args.sims} sims  vs  "
          f"Stockfish UCI_Elo {args.elo} at {args.sf_time}s/move", flush=True)

    seen = set()

    def opening():
        while True:
            b = chess.Board()
            for _ in range(8):
                moves = list(b.legal_moves)
                lg, _ = ev(encode_board(b)[None])
                l = lg[0, [move_to_index(m, b.turn) for m in moves]]
                b.push(rng.choices(moves, weights=np.exp(l - l.max()).tolist())[0])
            if b.outcome() is None and b.board_fen() not in seen:
                seen.add(b.board_fen())
                return b

    games = []
    for _ in range(args.pairs):
        start = opening()
        for we_white in (True, False):
            sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo, "Threads": 1})
            games.append({"board": start.copy(), "us": chess.WHITE if we_white else chess.BLACK,
                          "tree": MCTS(start.copy()), "sf": sf, "result": None})

    def finish(g):
        o = g["board"].outcome(claim_draw=True)
        if o is None and g["board"].ply() < MAX_PLIES:
            return
        w = o.winner if o else None
        g["result"] = 0.5 if w is None else float(w == g["us"])
        g["sf"].quit()
        print(f"game {games.index(g) + 1:2d} (we {'White' if g['us'] else 'Black'}): "
              f"{ {1.0: 'win', 0.5: 'draw', 0.0: 'loss'}[g['result']]} in "
              f"{g['board'].ply()} plies  [{time.time() - t0:.0f}s]", flush=True)

    def play(g, move):
        g["board"].push(move)
        g["tree"].advance(move)
        finish(g)

    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=len(games))
    while any(g["result"] is None for g in games):
        live = [g for g in games if g["result"] is None]
        ours = [g for g in live if g["board"].turn == g["us"]]
        theirs = [g for g in live if g["board"].turn != g["us"]]
        # Stockfish thinks in its own processes while we search on the GPU
        futures = {id(g): pool.submit(g["sf"].play, g["board"].copy(),
                                      chess.engine.Limit(time=args.sf_time))
                   for g in theirs}
        done = {id(g): 0 for g in ours}
        while any(done[id(g)] < args.sims for g in ours):
            batch, planes = [], []
            for g in ours:
                if done[id(g)] >= args.sims:
                    continue
                p, n = g["tree"].select_leaves(min(CFG.play_batch, args.sims - done[id(g)]))
                done[id(g)] += n
                if p:
                    batch.append((g, len(p)))
                    planes += p
            if planes:
                lg, vs = ev(np.stack(planes))
                i = 0
                for g, k in batch:
                    g["tree"].expand_leaves(lg[i:i + k], vs[i:i + k])
                    i += k
        for g in ours:
            play(g, g["tree"].best_move(0.0))
        for g in theirs:
            play(g, futures[id(g)].result().move)
    pool.shutdown()

    r = np.array([g["result"] for g in games])
    s = r.mean()
    sc = min(max(s, 1 / (2 * len(r))), 1 - 1 / (2 * len(r)))
    diff = 400 * math.log10(sc / (1 - sc))
    print(f"\nvs Stockfish {args.elo}: +{(r == 1).sum()} ={(r == 0.5).sum()} -{(r == 0).sum()}  "
          f"score {s:.2f}  ->  estimated Elo ~{args.elo + diff:.0f} ({diff:+.0f})  "
          f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
