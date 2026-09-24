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

A single game (engine.py, playing a human) batches *within* one tree instead:

    planes, n = mcts.select_leaves(k)    # up to k distinct leaves (virtual loss)
    mcts.expand_leaves(logits, values)   # answer them in the same order

Each descent adds a virtual loss to the edges it walks (as if that line had
just lost), steering the next descent in the same batch onto a different line.
Virtual losses live in a separate integer array (Node.VL), so reverting them
is exact and N/W only ever hold real results.

Conventions:
  - Node statistics live on edges (numpy arrays per node, vectorized PUCT).
  - Two implementations with one API: PyMCTS (this file, the reference) and
    NativeMCTS (a thin wrapper over native/fastmcts.cpp, the same algorithm
    in C++, ~10x faster). `MCTS` is the native one when the extension is
    built (native/build.sh) and CFG.native_mcts is on, else PyMCTS.
  - The network value is from the perspective of the side to move at a node;
    it is negated at every step while backing up.
  - The tree owns a private search board (a stack-less copy of the live
    board, kept in sync by advance()). Simulations push moves on the way down
    and pop them after the leaf is handled, instead of copying the board per
    simulation; legal moves are generated once per leaf and reused for both
    terminal detection and expansion.
  - Repetitions: a leaf whose position already occurred in the game or
    earlier on the current search line is scored as a draw (the usual
    in-search twofold rule), so the engine steers into repetitions when
    worse and avoids them when better. The game loop still ends games by
    threefold via board.outcome(claim_draw=True).
"""

import math

import chess
import numpy as np

from config import CFG
from encoding import encode_board, move_to_index


class Node:
    __slots__ = ("moves", "P", "N", "W", "children", "terminal", "in_flight",
                 "VL")

    def __init__(self):
        self.moves = None        # list[chess.Move] once expanded
        self.P = None            # priors          (k,) f32
        self.N = None            # visit counts    (k,) i32
        self.W = None            # total value     (k,) f32
        self.children = None     # list[Node|None]
        self.terminal = None     # cached value if the position is game-over
        self.in_flight = False   # selected in the current batch, awaiting eval
        self.VL = None           # virtual losses in flight per edge (k,) i32


def _terminal_value(board: chess.Board, moves: list) -> float | None:
    """Value for the side to move given its legal moves, or None if not over."""
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0.0
    return None


def _position_keys(board: chess.Board) -> set:
    """Repetition keys of every position in the board's game history."""
    b = board.copy()
    keys = {b._transposition_key()}
    while b.move_stack:
        b.pop()
        keys.add(b._transposition_key())
    return keys


_COLLISION = object()     # _descend(): leaf already awaiting evaluation


class PyMCTS:
    def __init__(self, board: chess.Board, add_noise: bool = False,
                 c_puct: float = CFG.c_puct,
                 fpu_reduction: float | None = CFG.fpu_reduction,
                 repetition_draws: bool = True):
        # fpu_reduction=None and repetition_draws=False give the original
        # AlphaZero rules (unvisited Q = 0, repetitions ignored) — used by the
        # benchmarks to compare against the pre-FPU baseline like for like.
        self.board = board                  # the live game board (shared)
        self._search = board.copy(stack=False)  # private push/pop search board
        self._history = _position_keys(board)   # positions seen in the game
        self.root = Node()
        self.add_noise = add_noise
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.repetition_draws = repetition_draws
        self._noise_pending = add_noise
        self._pending = None                # one leaf awaiting eval (select_leaf)
        self._batch = []                    # leaves awaiting eval (select_leaves)

    # -- one simulation, phase 1: descend ---------------------------------

    def select_leaf(self) -> np.ndarray | None:
        """
        Walk from root to a leaf. If the leaf is terminal, back it up and
        return None. Otherwise return its encoded planes; the caller must
        call expand_backup() with the network output before the next select.
        """
        assert self._pending is None and not self._batch, \
            "expand_backup()/expand_leaves() not called after select"
        leaf = self._descend(virtual_loss=False)
        if leaf is None:
            return None
        self._pending = leaf
        return leaf[4]

    # -- one simulation, phase 2: expand + back up -------------------------

    def expand_backup(self, policy_logits: np.ndarray, value: float) -> None:
        leaf, self._pending = self._pending, None
        self._expand(leaf, policy_logits, value, virtual_loss=False)

    # -- batched simulations within one tree (virtual loss) --------------------

    def select_leaves(self, k: int) -> tuple[list[np.ndarray], int]:
        """
        Run up to k simulations' descents. Returns (planes of the distinct
        leaves to evaluate, simulations used). Terminal leaves are backed up
        immediately and count as simulations; the batch stops early if a
        descent collides with a leaf already selected in it. Answer with
        expand_leaves() before selecting again.
        """
        assert self._pending is None and not self._batch, \
            "expand_backup()/expand_leaves() not called after select"
        planes, sims = [], 0
        for _ in range(k):
            leaf = self._descend(virtual_loss=True)
            if leaf is _COLLISION:
                break
            sims += 1
            if leaf is not None:
                self._batch.append(leaf)
                planes.append(leaf[4])
        return planes, sims

    def expand_leaves(self, policy_logits: np.ndarray, values: np.ndarray) -> None:
        batch, self._batch = self._batch, []
        for i, leaf in enumerate(batch):
            self._expand(leaf, policy_logits[i], float(values[i]), virtual_loss=True)

    def _descend(self, virtual_loss: bool):
        """
        Select down to a leaf on the push/pop search board, then pop back to
        the root. Returns (node, path, moves, turn, planes) for a leaf that
        needs the network, None for a terminal (already backed up), or
        _COLLISION if the leaf is already awaiting evaluation.
        """
        board = self._search
        node = self.root
        path = []
        line = [board._transposition_key()]     # positions on this search line

        while node.moves is not None:
            idx = self._puct_select(node)
            path.append((node, idx))
            if virtual_loss:
                if node.VL is None:
                    node.VL = np.zeros(len(node.moves), dtype=np.int32)
                node.VL[idx] += 1
            board.push(node.moves[idx])
            line.append(board._transposition_key())
            child = node.children[idx]
            if child is None:
                child = Node()
                node.children[idx] = child
            node = child

        if node.in_flight:
            self._unwind(len(path))
            self._revert_virtual_loss(path)
            return _COLLISION

        # Expanded nodes are never terminal, so only leaves need checking.
        term = node.terminal
        if term is None:
            key = line[-1]
            if path and self.repetition_draws and (
                    key in self._history or key in line[:-1]):
                term = 0.0                   # repetition: scored as a draw
            else:
                moves = list(board.legal_moves)
                term = _terminal_value(board, moves)
            node.terminal = term
        if term is not None:
            self._unwind(len(path))
            if virtual_loss:
                self._revert_virtual_loss(path)
            self._backup(path, term)
            return None

        leaf = (node, path, moves, board.turn, encode_board(board))
        self._unwind(len(path))
        node.in_flight = virtual_loss
        return leaf

    def _expand(self, leaf, policy_logits, value: float, virtual_loss: bool) -> None:
        node, path, moves, turn, _ = leaf
        if virtual_loss:
            self._revert_virtual_loss(path)
            node.in_flight = False
        idxs = [move_to_index(m, turn) for m in moves]
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
        N, W = node.N, node.W
        if node.VL is not None:          # in-flight visits count as losses
            N = N + node.VL
            W = W - node.VL
        n_total = N.sum()
        if n_total and self.fpu_reduction is not None:
            # First-play urgency (Lc0-style): score unvisited moves a bit below
            # this node's current average instead of at 0 (a "draw"), so a
            # losing position doesn't spray visits over every untried move and
            # a winning one still explores. The penalty grows with the share of
            # prior already explored.
            visited = N > 0
            fpu = (W.sum() / n_total
                   - self.fpu_reduction * math.sqrt(node.P[visited].sum()))
            q = np.where(visited, W / np.maximum(N, 1), fpu)
        else:                                # unvisited moves count as Q = 0
            q = W / np.maximum(N, 1)
        u = self.c_puct * node.P * (math.sqrt(n_total + 1) / (1 + N))
        return int(np.argmax(q + u))

    def _unwind(self, plies: int) -> None:
        """Pop the search board back to the root position."""
        for _ in range(plies):
            self._search.pop()

    @staticmethod
    def _revert_virtual_loss(path) -> None:
        for node, idx in path:
            node.VL[idx] -= 1

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

    def depth_stats(self) -> dict:
        """
        How deep the tree reaches: `depth` = length of the principal variation
        (most-visited child at each step, following nodes with >= 1 visit),
        `seldepth` = deepest expanded node, `mean_depth` = visit-weighted mean
        depth of expanded nodes.
        """
        depth, node = 0, self.root
        while node.moves is not None and node.N.sum() > 0:
            idx = int(np.argmax(node.N))
            child = node.children[idx]
            if child is None or child.moves is None:
                break
            depth, node = depth + 1, child
        seldepth, total, weighted = 0, 0, 0
        stack = [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            if node.moves is None:
                continue
            seldepth = max(seldepth, d)
            n = int(node.N.sum())
            total += n
            weighted += n * d
            stack.extend((c, d + 1) for c in node.children if c is not None)
        return {"depth": depth, "seldepth": seldepth,
                "mean_depth": weighted / total if total else 0.0}

    def pv(self, max_len: int = 64) -> list:
        """Principal variation: most-visited move at each step."""
        out, node = [], self.root
        while (len(out) < max_len and node is not None and node.moves is not None
               and node.N.sum() > 0):
            i = int(np.argmax(node.N))
            out.append(node.moves[i])
            node = node.children[i]
        return out

    def advance(self, move: chess.Move) -> None:
        """Play a move on the live board, reusing the subtree if we have it."""
        new_root = None
        if self.root.moves is not None and move in self.root.moves:
            new_root = self.root.children[self.root.moves.index(move)]
        self.root = new_root if new_root is not None else Node()
        # A leaf the search scored as a repetition draw can become the real
        # position (the game repeated once). At the root that isn't game over —
        # we still need a move — so drop the cached verdict and expand it.
        if self.root.moves is None:
            self.root.terminal = None
        self.board.push(move)
        self._search.push(move)
        self._history.add(self._search._transposition_key())
        if self.add_noise:
            if self.root.moves is not None:
                self._apply_noise()
            else:
                self._noise_pending = True


# -- native tree ---------------------------------------------------------------------

class _RootView:
    """Read-only snapshot of the native root with PyMCTS's Node fields."""
    __slots__ = ("moves", "N", "W", "P")

    def __init__(self, tree):
        if tree.root_expanded():
            self.moves = [chess.Move.from_uci(u) for u in tree.root_moves()]
            self.N, self.W, self.P = tree.root_N(), tree.root_W(), tree.root_P()
        else:
            self.moves = self.N = self.W = self.P = None


class NativeMCTS:
    """
    PyMCTS's API over the C++ tree (native/fastmcts.cpp). Same search, same
    rules; moves at a node are in a canonical order (from, to, promotion)
    rather than python-chess's, so ties can break differently. `root` is a
    snapshot (fresh arrays per access), not a live Node.
    """

    def __init__(self, board: chess.Board, add_noise: bool = False,
                 c_puct: float = CFG.c_puct,
                 fpu_reduction: float | None = CFG.fpu_reduction,
                 repetition_draws: bool = True):
        self.board = board                  # the live game board (shared)
        start = board.root()
        self._tree = fastmcts.Tree(
            start.fen(), [m.uci() for m in board.move_stack],
            -1 if board.ep_square is None else board.ep_square,
            c_puct, fpu_reduction, repetition_draws)
        self.add_noise = add_noise
        self._noise_pending = add_noise
        self._last_q = 0.0                  # Q of the last move chosen (contempt)
        if CFG.eval_cache:
            self._tree.set_cache(CFG.eval_cache)
        if CFG.solver:
            self._tree.set_solver(True)
        self.q_select = CFG.q_select        # >0: final move = best Q among moves
                                            # with >= this share of the top visits
        if (CFG.policy_temp, CFG.cpuct_factor) != (1.0, 0.0):
            self.set_search_params(CFG.policy_temp, CFG.cpuct_base, CFG.cpuct_factor)

    def update_contempt(self, contempt: float | None = None,
                        threshold: float | None = None) -> float:
        """Before a search: if we think we're better (root Q, or the last
        move's Q when the root is fresh, above `threshold`), score draws in
        the tree as -contempt for us, so a winning side steers away from
        repetitions and dead-drawn lines. Returns the contempt applied."""
        contempt = CFG.contempt if contempt is None else contempt
        threshold = CFG.contempt_threshold if threshold is None else threshold
        n, w = self._tree.root_N(), self._tree.root_W()
        q = float(w.sum() / n.sum()) if n.size and n.sum() > 0 else self._last_q
        c = contempt if (contempt > 0 and q > threshold) else 0.0
        self._tree.set_contempt(c)
        return c

    def set_solver(self, on: bool) -> None:
        """MCTS-solver: propagate proven mates up the tree."""
        self._tree.set_solver(bool(on))

    def set_cache(self, entries: int) -> None:
        """Network-output cache for transpositions (0 = off)."""
        self._tree.set_cache(int(entries))

    def cache_stats(self) -> tuple[int, int, int]:
        """(lookups, hits, entries)."""
        return self._tree.cache_stats()

    def set_root_fpu(self, value: float | None) -> None:
        """Absolute Q for unvisited moves at the root (None = the normal FPU)."""
        self._tree.set_root_fpu(float("nan") if value is None else value)

    def set_search_params(self, policy_temp: float = 1.0, cpuct_base: float = 38739.0,
                          cpuct_factor: float = 0.0) -> None:
        """Native-only knobs: prior softmax temperature and Lc0-style c_puct
        growth c(N) = c_puct + factor*ln((N+base)/base). Defaults = PyMCTS."""
        self._tree.set_search_params(policy_temp, cpuct_base, cpuct_factor)

    # -- search ----------------------------------------------------------------

    def select_leaf(self) -> np.ndarray | None:
        return self._tree.select_leaf()

    def expand_backup(self, policy_logits: np.ndarray, value: float) -> None:
        self._tree.expand_backup(policy_logits, float(value))
        self._after_expand()

    def select_leaves(self, k: int) -> tuple[list[np.ndarray], int]:
        """Like PyMCTS.select_leaves, but more batches may be selected before
        answering (pipelining); expand_leaves() answers the oldest."""
        planes, sims = self._tree.select_leaves(k)
        return list(planes), sims

    def pending_batches(self) -> int:
        return self._tree.pending_batches()

    def expand_leaves(self, policy_logits: np.ndarray, values: np.ndarray) -> None:
        self._tree.expand_leaves(policy_logits, values)
        self._after_expand()

    def _after_expand(self) -> None:
        if self._noise_pending and self._tree.root_expanded():
            self._apply_noise()

    def _apply_noise(self) -> None:
        self._noise_pending = False
        p = self._tree.root_P()
        noise = np.random.dirichlet([CFG.dirichlet_alpha] * len(p)).astype(np.float32)
        self._tree.set_root_P((1 - CFG.dirichlet_eps) * p + CFG.dirichlet_eps * noise)

    # -- results -------------------------------------------------------------------

    @property
    def root(self) -> _RootView:
        return _RootView(self._tree)

    def visit_counts(self):
        return ([chess.Move.from_uci(u) for u in self._tree.root_moves()],
                self._tree.root_N())

    def best_move(self, temperature: float = 0.0) -> chess.Move:
        moves, counts = self._tree.root_moves(), self._tree.root_N()
        if temperature <= 0:
            i = int(np.argmax(counts))
            w = self._tree.root_W()
            proven = np.array(self._tree.root_proven())
            if proven.size and (proven == 1).any():          # a proven win: take it
                wins = np.flatnonzero(proven == 1)
                i = int(wins[np.argmax(counts[wins])])
                self._last_q = 1.0
                return chess.Move.from_uci(moves[i])
            if proven.size and (proven == -1).any() and not (proven == -1).all():
                counts = np.where(proven == -1, -1, counts)  # never walk into a proven loss
                i = int(np.argmax(counts))
            if self.q_select > 0:
                q = w / np.maximum(counts, 1)
                ok = counts >= self.q_select * counts[i]
                i = int(np.flatnonzero(ok)[np.argmax(q[ok])])
            self._last_q = float(w[i] / max(counts[i], 1))
            return chess.Move.from_uci(moves[i])
        probs = counts.astype(np.float64) ** (1.0 / temperature)
        probs /= probs.sum()
        return chess.Move.from_uci(moves[int(np.random.choice(len(probs), p=probs))])

    def depth_stats(self) -> dict:
        return self._tree.depth_stats()

    def pv(self, max_len: int = 64) -> list:
        return [chess.Move.from_uci(u) for u in self._tree.pv(max_len)]

    def advance(self, move: chess.Move) -> None:
        self._tree.advance(move.uci())
        self.board.push(move)
        if self.add_noise:
            if self._tree.root_expanded():
                self._apply_noise()
            else:
                self._noise_pending = True


try:
    import fastmcts
except ImportError:                  # not built: native/build.sh
    fastmcts = None

MCTS = NativeMCTS if (fastmcts is not None and CFG.native_mcts) else PyMCTS
