"""
tune_match.py — fast A/B test of search settings: our engine vs itself.

Two configurations of the same net play each other at a fixed number of
simulations per move. All games run in lockstep and every tree's leaves go
into ONE GPU batch per round, so hundreds of games take minutes instead of
the hours a UCI match needs. Openings come from the 8-move suite; each is
played twice with colours swapped, so opening bias cancels.

    .venv/bin/python tune_match.py --b policy_temp=1.4
    .venv/bin/python tune_match.py --a play_batch=16 --b play_batch=32 --games 256
    .venv/bin/python tune_match.py --b c_puct=2.0,fpu=0.4 --nodes 1600

Settings (comma separated key=value): c_puct, fpu, policy_temp, cpuct_base,
cpuct_factor, play_batch, contempt, contempt_thr, pipeline. Unset keys use config.py. Different nets:
--a-model / --b-model.

Adjudication keeps games short: a side both engines agree is winning by
more than --win-q for 6 consecutive plies wins; after ply 80, |Q| < 0.05 for
both for 20 plies is a draw; max 400 plies.
"""

import argparse
import math
import os
import random
import time

import chess
import chess.pgn
import numpy as np
import torch

from config import CFG
from mcts import NativeMCTS
from network import Evaluator, load_checkpoint

KEYS = {"c_puct", "fpu", "policy_temp", "cpuct_base", "cpuct_factor", "play_batch",
        "contempt", "contempt_thr", "pipeline", "root_fpu", "qsel", "cache", "solver", "tm", "vscale"}


def parse(spec):
    out = {}
    for item in filter(None, (spec or "").split(",")):
        k, v = item.split("=")
        if k not in KEYS:
            raise SystemExit(f"unknown setting {k!r} (have {sorted(KEYS)})")
        out[k] = float(v)
    return out


def settings(over):
    return {"c_puct": over.get("c_puct", CFG.c_puct),
            "fpu": over.get("fpu", CFG.fpu_reduction),
            "policy_temp": over.get("policy_temp", CFG.policy_temp),
            "cpuct_base": over.get("cpuct_base", CFG.cpuct_base),
            "cpuct_factor": over.get("cpuct_factor", CFG.cpuct_factor),
            "play_batch": int(over.get("play_batch", CFG.play_batch)),
            "contempt": over.get("contempt", CFG.contempt),
            "contempt_thr": over.get("contempt_thr", CFG.contempt_threshold),
            # pipeline=1: each tree keeps one batch in flight while selecting the
            # next (as the pipelined UCI search does), so leaves are chosen with
            # two batches' worth of virtual loss outstanding
            "pipeline": int(over.get("pipeline", 0)),
            "root_fpu": over.get("root_fpu", None),     # absolute Q for unvisited root moves
            "qsel": over.get("qsel", CFG.q_select),
            "cache": int(over.get("cache", CFG.eval_cache)),
            "solver": int(over.get("solver", CFG.solver)),
            # tm=1 (with --bank): smart time use — stop early when the second
            # move can't catch the best in the remaining budget, think up to
            # 1.5x longer when the most-visited move isn't the best-scoring one
            "tm": int(over.get("tm", 0)),
            # multiply the net's value output (calibration: a net whose values
            # run hot searches as if c_puct were lower)
            "vscale": over.get("vscale", 1.0)}


def new_tree(board, s):
    t = NativeMCTS(board.copy(), c_puct=s["c_puct"], fpu_reduction=s["fpu"])
    t.set_search_params(s["policy_temp"], s["cpuct_base"], s["cpuct_factor"])
    t.set_root_fpu(s["root_fpu"])
    t.q_select = s["qsel"]
    t.set_cache(s["cache"])
    t.set_solver(s["solver"])
    return t


def openings(path, n, rng):
    """n distinct 8-move openings from the suite (reservoir sample)."""
    picked, seen = [], 0
    with open(path) as f:
        while (g := chess.pgn.read_game(f)) is not None:
            seen += 1
            if len(picked) < n:
                picked.append(g)
            elif rng.random() < n / seen:
                picked[rng.randrange(n)] = g
            if seen >= 20000:
                break
    boards = []
    for g in picked:
        b = g.board()
        for m in g.mainline_moves():
            b.push(m)
        boards.append(b)
    return boards


def elo(score, n):
    s = min(max(score, 0.5 / n), 1 - 0.5 / n)
    return -400 * math.log10(1 / s - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="", help="settings for engine A (baseline)")
    ap.add_argument("--b", default="", help="settings for engine B (candidate)")
    ap.add_argument("--a-model", default=(CFG.checkpoint if os.path.exists(CFG.checkpoint)
                                          else CFG.pretrained_weights))
    ap.add_argument("--b-model", default=None, help="default: same net as A")
    ap.add_argument("--games", type=int, default=128, help="even number")
    ap.add_argument("--nodes", type=int, default=800, help="simulations per move")
    ap.add_argument("--a-nodes", type=int, default=None, help="override --nodes for A")
    ap.add_argument("--b-nodes", type=int, default=None, help="override --nodes for B "
                    "(equal-time tests: scale by the measured speed ratio)")
    ap.add_argument("--win-q", type=float, default=0.9)
    ap.add_argument("--book", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "rating", "8moves_v3.pgn"),
                    help="opening suite (rating/setup.sh downloads it)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bank", type=int, default=0,
                    help="clock emulation: each side gets this many moves' worth of "
                         "--nodes per cycle (e.g. 40 = 40 moves in N*40 nodes, then "
                         "refilled) and spends bank/moves_left per move, like a UCI clock")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ma, _ = load_checkpoint(args.a_model, dev)
    eva = Evaluator(ma, dev)
    evb = eva
    if args.b_model:
        mb, _ = load_checkpoint(args.b_model, dev)
        evb = Evaluator(mb, dev)
    sa, sb = settings(parse(args.a)), settings(parse(args.b))
    sa["nodes"] = args.a_nodes or args.nodes
    sb["nodes"] = args.b_nodes or args.nodes
    print(f"A: {sa}\nB: {sb}\n{args.games} games"
          + (f"; A net {args.a_model}, B net {args.b_model}" if args.b_model else ""), flush=True)

    rng = random.Random(args.seed)
    starts = openings(args.book, args.games // 2, rng)
    games = []
    for i, b in enumerate(starts):
        for b_is_white in (False, True):
            games.append({"board": b.copy(), "b_white": b_is_white, "trees": None,
                          "bank": {"A": [args.bank * args.nodes, args.bank],
                                   "B": [args.bank * args.nodes, args.bank]},
                          "used": {"A": [0, 0], "B": [0, 0]},
                          "result": None, "streak": [0, 0, 0], "id": len(games)})
    for g in games:
        g["trees"] = {"A": new_tree(g["board"], sa), "B": new_tree(g["board"], sb)}

    def side_to_move(g):
        white = g["board"].turn == chess.WHITE
        return "B" if white == g["b_white"] else "A"

    def finish(g, white_score):
        g["result"] = white_score       # 1, 0.5, 0 from White's view

    t0 = time.time()
    done_games = 0
    while any(g["result"] is None for g in games):
        live = [g for g in games if g["result"] is None]
        todo = {g["id"]: 0 for g in live}
        for g in live:
            who = side_to_move(g)
            s = sa if who == "A" else sb
            g["trees"][who].update_contempt(s["contempt"], s["contempt_thr"])
        inflight = {}                    # pipelined trees: batch selected last round
        target, hard, fin = {}, {}, set()
        for g in live:
            who = side_to_move(g)
            s = sa if who == "A" else sb
            if args.bank:
                bank, left = g["bank"][who]
                target[g["id"]] = max(1, int(bank / max(left, 1)))
            else:
                target[g["id"]] = s["nodes"]
            hard[g["id"]] = int(target[g["id"]] * (1.5 if s["tm"] else 1.0))
        # one move for every live game: run each mover's search in lockstep
        while True:
            batch = {"A": [], "B": []}
            for g in live:
                who = side_to_move(g)
                s = sa if who == "A" else sb
                t = g["trees"][who]
                if g["id"] in fin or todo[g["id"]] >= hard[g["id"]]:
                    if g["id"] in inflight:              # drain the last pipelined batch
                        batch[who].append((t, inflight.pop(g["id"])))
                    continue
                planes, n = t.select_leaves(min(s["play_batch"], hard[g["id"]] - todo[g["id"]]))
                todo[g["id"]] += n
                if s["pipeline"]:
                    # evaluate LAST round's batch now; this one stays in flight
                    if g["id"] in inflight:
                        batch[who].append((t, inflight.pop(g["id"])))
                    if planes:
                        inflight[g["id"]] = planes
                elif planes:
                    batch[who].append((t, planes))
            if not batch["A"] and not batch["B"]:
                if not inflight and all(g["id"] in fin or todo[g["id"]] >= hard[g["id"]]
                                        for g in live):
                    break
                continue
            for who, ev in (("A", eva), ("B", evb)):
                if not batch[who]:
                    continue
                planes = np.stack([p for _, ps in batch[who] for p in ps])
                logits, values = ev(planes)
                vs = (sa if who == "A" else sb)["vscale"]
                if vs != 1.0:
                    values = np.clip(values * vs, -1.0, 1.0)
                i = 0
                for t, ps in batch[who]:
                    t.expand_leaves(logits[i:i + len(ps)], values[i:i + len(ps)])
                    i += len(ps)
            # smart time use: decide per tree whether to stop now
            for g in live:
                gid = g["id"]
                who = side_to_move(g)
                s = sa if who == "A" else sb
                if not s["tm"] or gid in fin or gid in inflight:
                    continue
                t = g["trees"][who]
                n = t._tree.root_N()
                if n.size < 2:
                    if n.size == 1 and todo[gid] >= 1:
                        fin.add(gid)                     # only one legal move
                    continue
                top = np.sort(n)[::-1]
                if top[0] - top[1] > max(target[gid] - todo[gid], 0):
                    fin.add(gid)                         # second move can't catch up
                elif todo[gid] >= target[gid]:
                    w = t._tree.root_W()
                    q = w / np.maximum(n, 1)
                    ok = n >= 0.1 * top[0]
                    if int(np.flatnonzero(ok)[np.argmax(q[ok])]) == int(np.argmax(n)):
                        fin.add(gid)                     # stable: stop at the budget
        # play the moves
        for g in live:
            who = side_to_move(g)
            if args.bank:
                bk = g["bank"][who]
                bk[0] -= todo[g["id"]]
                bk[1] -= 1
                if bk[1] <= 0:                       # new time-control cycle
                    bk[0] += args.bank * args.nodes
                    bk[1] = args.bank
            g["used"][who][0] += todo[g["id"]]
            g["used"][who][1] += 1
            t = g["trees"][who]
            root = t.root
            move = t.best_move(0.0)
            q = float(root.W[root.moves.index(move)] / max(root.N[root.moves.index(move)], 1))
            white_q = q if g["board"].turn == chess.WHITE else -q
            for tr in g["trees"].values():
                tr.advance(move)            # both trees share the game board
            b = g["board"]
            b.push(move)
            o = b.outcome(claim_draw=True)
            st = g["streak"]
            st[0] = st[0] + 1 if white_q > args.win_q else 0
            st[1] = st[1] + 1 if white_q < -args.win_q else 0
            st[2] = st[2] + 1 if abs(white_q) < 0.05 and b.ply() > 80 else 0
            if o is not None:
                finish(g, 0.5 if o.winner is None else float(o.winner == chess.WHITE))
            elif st[0] >= 6:
                finish(g, 1.0)
            elif st[1] >= 6:
                finish(g, 0.0)
            elif st[2] >= 20 or b.ply() >= 400:
                finish(g, 0.5)
            if g["result"] is not None:
                done_games += 1
                if done_games % 32 == 0 or done_games == len(games):
                    sc = [(g2["result"] if g2["b_white"] else 1 - g2["result"])
                          for g2 in games if g2["result"] is not None]
                    s = float(np.mean(sc))
                    print(f"  {done_games}/{len(games)} games: B scores {s:.3f} "
                          f"({elo(s, len(sc)):+.0f} Elo)  [{time.time() - t0:.0f}s]", flush=True)

    for who in ("A", "B"):
        tot = sum(g["used"][who][0] for g in games)
        mv = sum(g["used"][who][1] for g in games)
        print(f"{who}: {tot / max(mv, 1):,.0f} nodes/move on average")
    # B's score per game and per opening pair (pairs cancel opening bias)
    per = np.array([g["result"] if g["b_white"] else 1 - g["result"] for g in games])
    w, d, l = (per == 1).sum(), (per == 0.5).sum(), (per == 0).sum()
    pairs = per.reshape(-1, 2).sum(1)                   # 0 .. 2 per opening
    n = len(pairs)
    mean = pairs.mean() / 2
    se = pairs.std(ddof=1) / 2 / math.sqrt(n) if n > 1 else 0.5
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    print(f"\nB vs A: +{w} ={d} -{l}  score {mean:.3f}  ->  {elo(mean, len(per)):+.0f} Elo "
          f"(95% CI {elo(max(lo, 1e-3), len(per)):+.0f} .. {elo(min(hi, 1 - 1e-3), len(per)):+.0f})"
          f"  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
