"""
engine.py — alpha-beta search using the trained CNN evaluator.
Falls back to material eval if no model is found (same as before).
"""

import os

import chess
import chess.polyglot
import torch

from model import BoardEvaluator, board_to_tensor

MAX_DEPTH = 3        # NN eval is slower than material count; 3 is fast enough
WIN_SCORE = 10_000
BOOK_PATH = "baron30.bin"
MODEL_PATH = "evaluator.pt"

# ---------------------------------------------------------------------------
# Load model (once, at import time)
# ---------------------------------------------------------------------------

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: BoardEvaluator | None = None

def _load_model() -> BoardEvaluator | None:
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        m = BoardEvaluator()
        checkpoint = torch.load(MODEL_PATH, map_location=_device)
        m.load_state_dict(checkpoint["model"])
        m.to(_device)
        m.eval()
        print(f"Neural evaluator loaded from {MODEL_PATH}")
        return m
    except Exception as e:
        print(f"Could not load model ({e}); falling back to material eval")
        return None

_model = _load_model()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 1000,
}

def _material_score(board: chess.Board) -> int:
    score = 0
    for pt, val in PIECE_VALUES.items():
        score += len(board.pieces(pt, chess.WHITE)) * val
        score -= len(board.pieces(pt, chess.BLACK)) * val
    return score


def evaluate(board: chess.Board) -> float:
    if _model is not None:
        with torch.no_grad():
            t = board_to_tensor(board).unsqueeze(0).to(_device)  # (1,18,8,8)
            val = _model(t).item()                               # in (-1, 1)
        # Scale to centipawn-like range so WIN_SCORE comparisons work
        return val * 900
    return float(_material_score(board))


def terminal_score(board: chess.Board) -> float:
    if board.is_checkmate():
        return -WIN_SCORE if board.turn == chess.WHITE else WIN_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    return evaluate(board)


# ---------------------------------------------------------------------------
# Move ordering (MVV-LVA for captures → search best lines first)
# ---------------------------------------------------------------------------

def _move_priority(board: chess.Board, move: chess.Move) -> int:
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        a_val = PIECE_VALUES.get(attacker.piece_type, 9) if attacker else 9
        return 10 * v_val - a_val
    return 0


def _sorted_moves(board: chess.Board):
    return sorted(board.legal_moves,
                  key=lambda m: _move_priority(board, m),
                  reverse=True)


# ---------------------------------------------------------------------------
# Alpha-beta
# ---------------------------------------------------------------------------

def alphabeta(board: chess.Board, depth: int,
              alpha: float, beta: float, maximizing: bool) -> float:
    if depth == 0 or board.is_game_over():
        return terminal_score(board)

    if maximizing:
        best = float(-WIN_SCORE)
        for move in _sorted_moves(board):
            board.push(move)
            score = alphabeta(board, depth - 1, alpha, beta, False)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if beta <= alpha:
                break
        return best

    best = float(WIN_SCORE)
    for move in _sorted_moves(board):
        board.push(move)
        score = alphabeta(board, depth - 1, alpha, beta, True)
        board.pop()
        if score < best:
            best = score
        if best < beta:
            beta = best
        if beta <= alpha:
            break
    return best


# ---------------------------------------------------------------------------
# Opening book
# ---------------------------------------------------------------------------

def book_move(board: chess.Board) -> chess.Move | None:
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            entry = reader.get(board)
            if entry:
                return entry.move
    except FileNotFoundError:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def choose_move(board: chess.Board) -> chess.Move | None:
    move = book_move(board)
    if move is not None:
        print(f"Book move: {move.uci()}")
        return move

    maximizing = board.turn == chess.WHITE
    best_move = None
    best_score = float(-WIN_SCORE) if maximizing else float(WIN_SCORE)

    for move in _sorted_moves(board):
        board.push(move)
        score = alphabeta(board, MAX_DEPTH - 1,
                          float(-WIN_SCORE), float(WIN_SCORE), not maximizing)
        board.pop()

        if maximizing and score > best_score:
            best_score = score
            best_move = move
        elif not maximizing and score < best_score:
            best_score = score
            best_move = move

    return best_move
