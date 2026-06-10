"""
search.py — Stage 1: classical game-tree search with a material evaluator.

Two searches over the same negamax formulation (scores are always from the
side to move's point of view, so one function handles both colors):

  minimax    visits every node to a fixed depth
  alphabeta  same result, but prunes branches that provably can't change the
             choice at the root; captures are searched first (MVV-LVA) so
             good moves raise alpha early and more of the tree gets cut

Both use make/unmake traversal on a single board (board.push / board.pop)
rather than copying the position at every node.

    python search.py            # compare the two on a few test positions
"""

import time

import chess

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
MATE = 100_000


def evaluate(board: chess.Board) -> int:
    """Material balance in centipawns, from the side to move's perspective."""
    score = 0
    for pt, val in PIECE_VALUES.items():
        score += val * (len(board.pieces(pt, chess.WHITE))
                        - len(board.pieces(pt, chess.BLACK)))
    return score if board.turn == chess.WHITE else -score


def _terminal(board: chess.Board, ply: int) -> int | None:
    if board.is_checkmate():
        return -MATE + ply            # prefer faster mates / slower losses
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return None


class Searcher:
    def __init__(self):
        self.nodes = 0

    # -- plain minimax --------------------------------------------------------

    def minimax(self, board: chess.Board, depth: int, ply: int = 0) -> int:
        self.nodes += 1
        term = _terminal(board, ply)
        if term is not None:
            return term
        if depth == 0:
            return evaluate(board)
        best = -MATE
        for move in board.legal_moves:
            board.push(move)
            best = max(best, -self.minimax(board, depth - 1, ply + 1))
            board.pop()
        return best

    # -- alpha-beta -------------------------------------------------------------

    def alphabeta(self, board: chess.Board, depth: int,
                  alpha: int = -MATE, beta: int = MATE, ply: int = 0) -> int:
        self.nodes += 1
        term = _terminal(board, ply)
        if term is not None:
            return term
        if depth == 0:
            return evaluate(board)
        for move in ordered_moves(board):
            board.push(move)
            score = -self.alphabeta(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta           # opponent won't allow this line: cut
            alpha = max(alpha, score)
        return alpha


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    """Captures first, most valuable victim / least valuable attacker."""
    def key(move):
        if not board.is_capture(move):
            return 0
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v = PIECE_VALUES[victim.piece_type] if victim else PIECE_VALUES[chess.PAWN]
        return 10 * v - PIECE_VALUES[attacker.piece_type]
    return sorted(board.legal_moves, key=key, reverse=True)


def best_move(board: chess.Board, depth: int, algorithm: str = "alphabeta"):
    """Returns (move, score, nodes searched)."""
    s = Searcher()
    best, best_score = None, -MATE - 1
    alpha = -MATE
    moves = ordered_moves(board) if algorithm == "alphabeta" else board.legal_moves
    for move in moves:
        board.push(move)
        if algorithm == "alphabeta":
            score = -s.alphabeta(board, depth - 1, -MATE, -alpha, ply=1)
        else:
            score = -s.minimax(board, depth - 1, ply=1)
        board.pop()
        if score > best_score:
            best, best_score = move, score
            alpha = max(alpha, score)
    return best, best_score, s.nodes


# -- demo ----------------------------------------------------------------------

POSITIONS = {
    "start position": chess.STARTING_FEN,
    "hanging queen": "rnb1kbnr/pppp1ppp/8/4p1q1/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3",
    "back-rank mate": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "middlegame": "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 0 8",
}


def main():
    depth = 3
    print(f"depth {depth}: minimax vs alpha-beta (same evaluator)\n")
    print(f"{'position':16s} {'algorithm':10s} {'move':6s} {'score':>7s} "
          f"{'nodes':>9s} {'time':>7s}")
    for name, fen in POSITIONS.items():
        board = chess.Board(fen)
        results = {}
        for algo in ("minimax", "alphabeta"):
            t0 = time.perf_counter()
            move, score, nodes = best_move(board, depth, algo)
            dt = time.perf_counter() - t0
            results[algo] = (score, nodes)
            print(f"{name:16s} {algo:10s} {board.san(move):6s} {score:7d} "
                  f"{nodes:9,d} {dt:6.2f}s")
        assert results["minimax"][0] == results["alphabeta"][0], "scores differ"
        ratio = results["minimax"][1] / results["alphabeta"][1]
        print(f"{'':16s} -> same score, alpha-beta searched {ratio:.0f}x fewer nodes\n")


if __name__ == "__main__":
    main()
