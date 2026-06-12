"""
mcts.py — PUCT Monte-Carlo Tree Search with externalized (batchable) evaluation.

The tree never calls the network itself. Instead the driver loop does:

    planes = mcts.select_leaf()          # descend to an unevaluated leaf
    if planes is not None:               # None => terminal, backed up already
        logits, value = evaluator(batch_of_planes)   # batched across games!
        mcts.expand_backup(logits, value)

This lets self-play run N games in lockstep and evaluate all their leaves in
a single GPU call, which is where all the throughput on a single GPU comes
from.

Conventions:
  - Node statistics live on edges (numpy arrays per node, vectorized PUCT).
  - The network value is from the perspective of the side to move at a node;
    it is negated at every step while backing up.
  - Terminal detection inside the search ignores threefold repetition (the
    move stack isn't copied, for speed). The *game* loop still ends games by
    repetition via board.outcome(claim_draw=True).
"""

import math

import chess
import numpy as np

from config import CFG
from encoding import encode_board, move_to_index


class Node:
    __slots__ = ("moves", "P", "N", "W", "children")

    def __init__(self):
        self.moves = None        # list[chess.Move] once expanded
        self.P = None            # priors          (k,) f32
        self.N = None            # visit counts    (k,) i32
        self.W = None            # total value     (k,) f32
        self.children = None     # list[Node|None]


def _terminal_value(board: chess.Board) -> float | None:
    """Value from the perspective of the side to move, or None if not over."""
    if board.is_checkmate():
        return -1.0
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.halfmove_clock >= 100):
        return 0.0
    return None


class MCTS:
    def __init__(self, board: chess.Board, add_noise: bool = False,
                 c_puct: float = CFG.c_puct):
        self.board = board                  # the live game board (shared)
        self.root = Node()
        self.add_noise = add_noise
        self.c_puct = c_puct
        self._noise_pending = add_noise
        self._pending = None                # (node, path, leaf_board) awaiting eval

    # -- one simulation, phase 1: descend ---------------------------------

    def select_leaf(self) -> np.ndarray | None:
        """
        Walk from root to a leaf. If the leaf is terminal, back it up and
        return None. Otherwise return its encoded planes; the caller must
        call expand_backup() with the network output before the next select.
        """
        assert self._pending is None, "expand_backup() not called after select_leaf()"
        board = self.board.copy(stack=False)
        node = self.root
        path = []

        while True:
            term = _terminal_value(board)
            if term is not None:
                self._backup(path, term)
                return None
            if node.moves is None:
                self._pending = (node, path, board)
                return encode_board(board)

            idx = self._puct_select(node)
            path.append((node, idx))
            board.push(node.moves[idx])
            child = node.children[idx]
            if child is None:
                child = Node()
                node.children[idx] = child
            node = child

    # -- one simulation, phase 2: expand + back up -------------------------

    def expand_backup(self, policy_logits: np.ndarray, value: float) -> None:
        node, path, board = self._pending
        self._pending = None

        moves = list(board.legal_moves)
        idxs = [move_to_index(m, board.turn) for m in moves]
        logits = policy_logits[idxs]
        logits -= logits.max()
        priors = np.exp(logits)
        priors /= priors.sum()

        node.moves = moves
        node.P = priors.astype(np.float32)
        node.N = np.zeros(len(moves), dtype=np.int32)
        node.W = np.zeros(len(moves), dtype=np.float32)
        node.children = [None] * len(moves)

        if node is self.root and self._noise_pending:
            self._apply_noise()
        self._backup(path, float(value))

    # -- internals ----------------------------------------------------------

    def _puct_select(self, node: Node) -> int:
        q = node.W / np.maximum(node.N, 1)
        u = self.c_puct * node.P * (math.sqrt(node.N.sum() + 1) / (1 + node.N))
        return int(np.argmax(q + u))

    @staticmethod
    def _backup(path, leaf_value: float) -> None:
        v = leaf_value
        for node, idx in reversed(path):
            v = -v                       # parent's side made this move
            node.N[idx] += 1
            node.W[idx] += v

    def _apply_noise(self) -> None:
        self._noise_pending = False
        k = len(self.root.moves)
        noise = np.random.dirichlet([CFG.dirichlet_alpha] * k).astype(np.float32)
        self.root.P = (1 - CFG.dirichlet_eps) * self.root.P + CFG.dirichlet_eps * noise

    # -- driving the game ----------------------------------------------------

    def visit_counts(self):
        """(moves, counts) at the root — the training policy target."""
        return self.root.moves, self.root.N

    def best_move(self, temperature: float = 0.0) -> chess.Move:
        counts = self.root.N
        if temperature <= 0:
            return self.root.moves[int(np.argmax(counts))]
        probs = counts.astype(np.float64) ** (1.0 / temperature)
        probs /= probs.sum()
        return self.root.moves[int(np.random.choice(len(probs), p=probs))]

    def advance(self, move: chess.Move) -> None:
        """Play a move on the live board, reusing the subtree if we have it."""
        new_root = None
        if self.root.moves is not None and move in self.root.moves:
            new_root = self.root.children[self.root.moves.index(move)]
        self.root = new_root if new_root is not None else Node()
        self.board.push(move)
        if self.add_noise:
            if self.root.moves is not None:
                self._apply_noise()
            else:
                self._noise_pending = True
