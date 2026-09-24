"""
uci.py — UCI protocol front end, so match runners (fastchess, cutechess-cli)
and chess GUIs can play the engine.

Search is time-based: from the clock (wtime/btime/winc/binc/movestogo, or
movetime) it budgets a share of the remaining time and runs batched MCTS
(virtual loss, tree reused between moves) until the budget or `stop`.
`go nodes N` runs N simulations instead.

Defaults suit engine-vs-engine rating matches (CCRL style): the engine's own
opening book and the online endgame tablebase are off (the match supplies the
openings; CCRL allows only local 3-5 piece tablebases). Both are UCI options.

    ./uci_engine.sh            # the executable to register in a GUI / runner

Every move is also published to /tmp/azchess_live/ for live_viewer.py.
"""

import contextlib
import json
import math
import os
import sys
import threading
import time

import chess
import numpy as np

NAME = "AlphaZeroChess 128x10"
LIVE_DIR = "/tmp/azchess_live"   # per-engine status files for live_viewer.py
MOVE_OVERHEAD = 0.05          # seconds kept back per move for I/O and latency


_OUT = sys.stdout             # the real stdout, captured before any redirect


def send(line: str) -> None:
    _OUT.write(line + "\n")
    _OUT.flush()


def q_to_cp(q: float) -> int:
    """Win-probability-style value in (-1, 1) -> centipawns (Lc0's mapping)."""
    q = max(-0.999, min(0.999, q))
    return int(round(111.714640912 * math.tan(1.5620688421 * q)))


def write_live(board, color, key=None, **extra):
    """Publish a game's current state for live_viewer.py (best effort).
    key names the game; default is this process's pid (one game per engine)."""
    try:
        os.makedirs(LIVE_DIR, exist_ok=True)
        path = os.path.join(LIVE_DIR, f"{key or os.getpid()}.json")
        state = {"fen": board.fen(), "color": color, "time": time.time(),
                 "last": board.peek().uci() if board.move_stack else None,
                 "ply": board.ply(), **extra}
        with open(path + ".tmp", "w") as f:
            json.dump(state, f)
        os.replace(path + ".tmp", path)
    except Exception:
        pass


def clear_live(key=None):
    try:
        os.remove(os.path.join(LIVE_DIR, f"{key or os.getpid()}.json"))
    except OSError:
        pass


class Engine:
    def __init__(self):
        self.board = chess.Board()
        self.options = {"OwnBook": False, "Tablebase": False}
        self.engine = None                   # the engine module, loaded lazily
        self.loaded = threading.Event()
        self.stop = threading.Event()
        self.search_thread = None
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        # engine.py prints to stdout on import; keep stdout clean for UCI.
        # (redirect_stdout is process-wide, which is why send() writes to _OUT.)
        with contextlib.redirect_stdout(sys.stderr):
            import engine
        # Capture the CUDA graphs for every batch size search can use now, so
        # the one-time capture cost never lands on a move's clock.
        if engine._evaluator is not None:
            b = 1
            while b <= engine.CFG.play_batch:
                engine._evaluator(np.zeros((b, 19, 8, 8), dtype=np.uint8))
                b *= 2
        self.engine = engine
        self.loaded.set()

    # -- commands -------------------------------------------------------------

    def finish_search(self):
        """Stop and wait for a running search (a GUI should send `stop` first,
        but a new position/go mid-search must never touch a busy tree)."""
        if self.search_thread is not None and self.search_thread.is_alive():
            self.stop.set()
            self.search_thread.join()

    def position(self, tokens):
        self.finish_search()
        if tokens and tokens[0] == "startpos":
            board, rest = chess.Board(), tokens[1:]
        elif tokens and tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            board, rest = chess.Board(fen), tokens[7:]
        else:
            return
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                board.push_uci(uci)
        self.board = board

    def go(self, tokens):
        args, i = {}, 0
        while i < len(tokens):
            key = tokens[i]
            if key in ("wtime", "btime", "winc", "binc", "movestogo",
                       "movetime", "nodes", "depth"):
                args[key] = int(tokens[i + 1])
                i += 2
            else:                            # infinite, ponder, ...
                args[key] = True
                i += 1
        self.loaded.wait()
        self.finish_search()
        self.stop.clear()
        self.search_thread = threading.Thread(target=self._search, args=(args,),
                                              daemon=True)
        self.search_thread.start()

    def _budget(self, args):
        """Seconds to think, or None for no time limit."""
        if "movetime" in args:
            return max(0.01, args["movetime"] / 1000 - MOVE_OVERHEAD)
        if args.get("infinite") or "nodes" in args:
            return None
        white = self.board.turn == chess.WHITE
        left = args.get("wtime" if white else "btime")
        if left is None:
            return 1.0
        inc = args.get("winc" if white else "binc", 0) / 1000
        left /= 1000
        moves_to_go = args.get("movestogo", 30)
        budget = left / max(moves_to_go, 1) + 0.8 * inc
        return max(0.01, min(budget, left * 0.5) - MOVE_OVERHEAD)

    def _search(self, args):
        e, board = self.engine, self.board
        cfg = e.CFG
        if board.is_game_over():
            send("bestmove 0000")
            return
        if self.options["OwnBook"]:
            move = e.book_move(board)
            if move is not None:
                send(f"bestmove {move.uci()}")
                return
        if self.options["Tablebase"]:
            move = e.tablebase_move(board)
            if move is not None:
                send(f"bestmove {move.uci()}")
                return

        t0 = time.time()
        write_live(board, "white" if board.turn == chess.WHITE else "black",
                   thinking=True)                # opponent's move is in; we're up
        budget = self._budget(args)
        deadline = None if budget is None else t0 + budget
        max_sims = args.get("nodes")
        tree = e._tree_for(board)
        done, last_info, only_move = 0, t0, None
        while not self.stop.is_set():
            if deadline is not None and time.time() >= deadline:
                break
            if max_sims is not None and done >= max_sims:
                break
            k = cfg.play_batch if max_sims is None else min(cfg.play_batch, max_sims - done)
            planes, n = tree.select_leaves(k)
            done += n
            if planes:
                logits, values = e._evaluator(np.stack(planes))
                tree.expand_leaves(logits, values)
            if time.time() - last_info > 1.0:
                self._info(tree, done, t0)
                last_info = time.time()
            if only_move is None and done >= 1:
                only_move = len(tree.root.moves or ()) == 1
            if only_move:
                break                        # only one legal move: play it
        self._info(tree, done, t0)
        best = tree.best_move(temperature=0.0)
        send(f"bestmove {best.uci()}")
        root = tree.root
        i = root.moves.index(best)
        after = board.copy()
        after.push(best)
        write_live(after, "white" if board.turn == chess.WHITE else "black",
                   eval_cp=q_to_cp(root.W[i] / max(root.N[i], 1)),
                   depth=tree.depth_stats()["depth"], sims=done)

    def _info(self, tree, sims, t0):
        root = tree.root
        if root.N is None or root.N.sum() == 0:
            return
        best = int(np.argmax(root.N))
        q = root.W[best] / root.N[best]
        stats = tree.depth_stats()
        pv = [m.uci() for m in tree.pv(20)]
        elapsed = max(time.time() - t0, 1e-3)
        send(f"info depth {stats['depth']} seldepth {stats['seldepth']} "
             f"nodes {sims} nps {int(sims / elapsed)} time {int(elapsed * 1000)} "
             f"score cp {q_to_cp(q)} pv {' '.join(pv)}")


def main():
    eng = Engine()
    for raw in sys.stdin:
        tokens = raw.strip().split()
        if not tokens:
            continue
        cmd, rest = tokens[0], tokens[1:]
        if cmd == "uci":
            send(f"id name {NAME}")
            send("id author Peter")
            send("option name OwnBook type check default false")
            send("option name Tablebase type check default false")
            send("uciok")
        elif cmd == "isready":
            eng.loaded.wait()
            send("readyok")
        elif cmd == "setoption" and "name" in rest and "value" in rest:
            name = " ".join(rest[rest.index("name") + 1:rest.index("value")])
            value = rest[rest.index("value") + 1].lower() == "true"
            if name in eng.options:
                eng.options[name] = value
        elif cmd == "ucinewgame":
            eng.loaded.wait()
            eng.finish_search()
            eng.engine._tree = None          # don't reuse a tree across games
        elif cmd == "position":
            eng.position(rest)
        elif cmd == "go":
            eng.go(rest)
        elif cmd == "stop":
            eng.finish_search()
        elif cmd == "quit":
            eng.stop.set()
            clear_live()
            break


if __name__ == "__main__":
    main()
