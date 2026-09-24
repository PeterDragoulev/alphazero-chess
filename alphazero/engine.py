"""
engine.py — play interface: opening book → MCTS with the trained net.

Exposes choose_move(board), MODEL_PATH and _model (ui.py and match_vs_ab.py
import them). Loads $CHESS_MODEL if set, else data/checkpoint.pt if you've trained one,
else the shipped pretrained weights; with neither, falls back to a shallow material
alpha-beta so the UI stays playable.
"""

import json
import os
import urllib.parse
import urllib.request

import chess
import chess.polyglot
import numpy as np
import torch

from config import CFG
from mcts import MCTS
from network import Evaluator, load_checkpoint

# CHESS_MODEL=path/to.pt picks a specific checkpoint (e.g. to compare nets);
# otherwise your trained checkpoint, else the shipped pretrained weights.
MODEL_PATH = os.environ.get("CHESS_MODEL") or (
    CFG.checkpoint if os.path.exists(CFG.checkpoint) else CFG.pretrained_weights)

# Online endgame tablebase (Lichess, up to 7 pieces). No local storage — probed
# over HTTP only when the board is already simple, so it adds no startup cost
# and falls back to search on any failure (offline, >7 pieces, timeout).
TABLEBASE_URL = "https://tablebase.lichess.ovh/standard"
TABLEBASE_MAX_PIECES = 7
TABLEBASE_TIMEOUT = 5.0

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_evaluator = None
_evaluator2 = None
if os.path.exists(MODEL_PATH):
    try:
        _model, _ckpt = load_checkpoint(MODEL_PATH, _device)
        _model.eval()
        _evaluator = Evaluator(_model, _device, frozen=True)   # weights never change here
        _evaluator2 = Evaluator(_model, _device, frozen=True)  # 2nd in-flight batch (pipelining)
        print(f"Loaded {MODEL_PATH} ({_ckpt.get('games', 0):,} games trained)")
    except Exception as e:
        print(f"Could not load model ({e}); falling back to material eval")
else:
    print(f"No checkpoint at {MODEL_PATH}; falling back to material eval")


def book_move(board: chess.Board) -> chess.Move | None:
    """A book move, chosen by weight among the strong candidates (weight at
    least half the best) so repeated games don't replay one opening."""
    try:
        with chess.polyglot.open_reader(CFG.book_path) as reader:
            entries = list(reader.find_all(board))
    except FileNotFoundError:
        return None
    if not entries:
        return None
    top = max(e.weight for e in entries)
    strong = [e for e in entries if e.weight * 2 >= top]
    weights = np.array([e.weight for e in strong], dtype=np.float64)
    return strong[np.random.choice(len(strong), p=weights / weights.sum())].move


def tablebase_move(board: chess.Board) -> chess.Move | None:
    """Provably-optimal endgame move from the Lichess 7-piece tablebase.

    Returns None when the position has >7 pieces or the probe fails for any
    reason (no network, timeout, bad response) so the caller falls back to
    search. The API returns legal moves ordered best-first for the side to
    move, so the first one is optimal (win in fewest, or best draw/loss).
    """
    if chess.popcount(board.occupied) > TABLEBASE_MAX_PIECES:
        return None
    try:
        url = TABLEBASE_URL + "?" + urllib.parse.urlencode({"fen": board.fen()})
        with urllib.request.urlopen(url, timeout=TABLEBASE_TIMEOUT) as resp:
            moves = json.load(resp).get("moves")
        if moves:
            return chess.Move.from_uci(moves[0]["uci"])
    except Exception:
        return None
    return None


def _phase_simulations(board: chess.Board, base: int) -> int:
    """Spend more search where the tree is narrow. Endgames have few legal
    moves, so extra sims are nearly free in wall-clock yet buy the deep, exact
    lines (conversion, opposition, zugzwang) that decide them."""
    pieces = chess.popcount(board.occupied)
    if pieces <= 8:
        return base * 4
    if pieces <= 12:
        return base * 2
    return base


def choose_move(board: chess.Board,
                simulations: int | None = None,
                use_tablebase: bool = True) -> chess.Move | None:
    """Pick a move: opening book → endgame tablebase → MCTS (material fallback).

    simulations=None scales the search by game phase (more in the endgame);
    pass an int to force a fixed count. use_tablebase=False disables the
    online probe (used to reproduce the pre-tablebase engine for comparison).
    """
    if board.is_game_over():
        return None
    move = book_move(board)
    if move is not None:
        return move
    if use_tablebase:
        tb = tablebase_move(board)
        if tb is not None:
            return tb
    if _model is None:
        return _material_move(board)

    sims = (simulations if simulations is not None
            else _phase_simulations(board, CFG.play_simulations))
    tree = _tree_for(board)
    reused = int(tree.root.N.sum()) if tree.root.N is not None else 0
    done = run_search(tree, sims=sims)
    last_search.update(sims=done, reused=reused, **tree.depth_stats())
    return tree.best_move(temperature=0.0)


def run_search(tree, sims=None, deadline=None, stop=None, on_progress=None,
               hard_deadline=None) -> int:
    """
    Batched search on one tree until `sims` simulations, time.time() >=
    `deadline`, or stop.is_set(). Returns the simulations run.

    With CFG.play_pipeline (native tree, CUDA) two batches are in flight:
    while the GPU evaluates one, the CPU selects the next, so neither waits
    for the other. The second batch is chosen with the first's virtual
    losses still outstanding, like one batch of twice the size.

    Smart time use (CFG.smart_time, native tree, with a deadline): stop as
    soon as the second-most-visited root move can't catch the first in the
    time left at the current speed, and when `deadline` arrives with the
    most-visited move not the best-scoring one, keep going until
    `hard_deadline` (or until they agree).
    """
    import collections
    import time as _time
    if hasattr(tree, "update_contempt"):
        tree.update_contempt()
    pipelined = (CFG.play_pipeline and hasattr(tree, "pending_batches")
                 and _evaluator2 is not None)
    evs = (_evaluator, _evaluator2)
    smart = CFG.smart_time and hasattr(tree, "pending_batches")
    t_start = _time.time()
    pending = collections.deque()           # (evaluator, handle), oldest first
    done = turn = 0
    only_move = None
    while True:
        if stop is not None and stop.is_set():
            break
        now = _time.time()
        if deadline is not None and now >= deadline:
            if not (smart and hard_deadline is not None and now < hard_deadline
                    and _unstable(tree)):
                break
        if smart and done >= 256 and not pending and deadline is not None:
            n = tree._tree.root_N()
            if n.size >= 2:
                top = np.sort(n)[::-1]
                rate = done / max(now - t_start, 1e-3)
                if top[0] - top[1] > rate * max(deadline - now, 0):
                    break                            # the best move can't be overtaken
        if sims is not None and done >= sims:
            break
        k = CFG.play_batch if sims is None else min(CFG.play_batch, sims - done)
        planes, n = tree.select_leaves(k)
        done += n
        if not pipelined:
            if planes:
                logits, values = _evaluator(np.stack(planes))
                tree.expand_leaves(logits, values)
        else:
            if planes:
                ev = evs[turn % 2]
                pending.append((ev, ev.submit(np.stack(planes))))
                turn += 1
            if len(pending) >= 2 or (pending and not planes):
                ev, h = pending.popleft()
                tree.expand_leaves(*ev.fetch(h))
        if on_progress is not None:
            on_progress(done)
        if only_move is None and done >= 1 and not pending:
            only_move = len(tree.root.moves or ()) == 1
        if only_move:
            break                            # only one legal move: play it
    while pending:
        ev, h = pending.popleft()
        tree.expand_leaves(*ev.fetch(h))
    return done


def _unstable(tree) -> bool:
    """Most-visited root move isn't the best-scoring one (among moves with at
    least 10% of the top visits)."""
    n = tree._tree.root_N()
    if n.size < 2:
        return False
    q = tree._tree.root_W() / np.maximum(n, 1)
    ok = n >= 0.1 * n.max()
    return int(np.flatnonzero(ok)[np.argmax(q[ok])]) != int(np.argmax(n))


# Search tree kept between moves: when the next call continues the same game
# (our move + the opponent's reply), the matching subtree is reused instead of
# thrown away. Any other position (new game, takeback, setup) starts fresh.
_tree: MCTS | None = None
last_search: dict = {}          # stats of the most recent search (for the UI)


def _tree_for(board: chess.Board) -> MCTS:
    global _tree
    t = _tree
    if t is not None:
        played, stack = t.board.move_stack, board.move_stack
        if (len(played) <= len(stack) <= len(played) + 4
                and stack[:len(played)] == played):
            for move in stack[len(played):]:
                t.advance(move)
            if t.board.fen() == board.fen():
                return t
    _tree = MCTS(board.copy(), add_noise=False)   # never mutate the caller's board
    return _tree


# -- fallback: material alpha-beta (pre-training only) --------------------------

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_WIN = 10_000


def _material(board: chess.Board) -> int:
    score = 0
    for pt, val in _PIECE_VALUES.items():
        score += val * (len(board.pieces(pt, chess.WHITE))
                        - len(board.pieces(pt, chess.BLACK)))
    return score if board.turn == chess.WHITE else -score


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    if board.is_checkmate():
        return -_WIN
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth == 0:
        return _material(board)
    for move in board.legal_moves:
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return alpha


def _material_move(board: chess.Board, depth: int = 3) -> chess.Move | None:
    best_move, best = None, -_WIN - 1
    for move in board.legal_moves:
        board.push(move)
        score = -_negamax(board, depth - 1, -_WIN, _WIN)
        board.pop()
        if score > best:
            best, best_move = score, move
    return best_move
