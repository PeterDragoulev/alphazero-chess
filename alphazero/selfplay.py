"""
selfplay.py — parallel self-play that feeds the replay buffer.

CFG.parallel_games games run in lockstep: every game descends its tree to one
leaf, all leaves are evaluated in a single GPU call, and this repeats
CFG.simulations times per move. Then every game plays its move simultaneously.
Finished games are immediately replaced with fresh ones so the GPU batch
stays full.

A game contributes to the buffer only when it finishes (the value target z is
the game outcome). Stopping mid-game loses at most the in-flight games —
a few minutes of work.
"""

import chess
import numpy as np

from config import CFG
from encoding import encode_board, move_to_index
from mcts import MCTS
from network import Evaluator
from replay_buffer import ReplayBuffer


class _Game:
    __slots__ = ("board", "mcts", "records", "plies")

    def __init__(self):
        self.board = chess.Board()
        self.mcts = MCTS(self.board, add_noise=True)
        self.records = []        # (planes u8, pol_idx u16, pol_prob f16, turn)
        self.plies = 0


class SelfPlayPool:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.games = [_Game() for _ in range(CFG.parallel_games)]
        self.results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    def play_block(self, buffer: ReplayBuffer, target_games: int,
                   stop=None) -> tuple[int, int]:
        """
        Run move cycles until `target_games` games finish (or stop.requested).
        Returns (positions_added, games_finished).
        """
        positions = 0
        finished = 0
        while finished < target_games:
            if stop is not None and stop.requested:
                break
            self._run_simulations()
            for i, g in enumerate(self.games):
                done, n_pos = self._play_move(g, buffer)
                if done:
                    finished += 1
                    positions += n_pos
                    self.games[i] = _Game()
        return positions, finished

    # -- one move for all games ---------------------------------------------

    def _run_simulations(self) -> None:
        for _ in range(CFG.simulations):
            pending, planes = [], []
            for g in self.games:
                p = g.mcts.select_leaf()
                if p is not None:
                    pending.append(g)
                    planes.append(p)
            if pending:
                logits, values = self.evaluator(np.stack(planes))
                for i, g in enumerate(pending):
                    g.mcts.expand_backup(logits[i], float(values[i]))

    def _play_move(self, g: _Game, buffer: ReplayBuffer) -> tuple[bool, int]:
        """Record the search result, play a move. Returns (game_over, n_pos)."""
        moves, counts = g.mcts.visit_counts()
        idx = np.array([move_to_index(m, g.board.turn) for m in moves],
                       dtype=np.uint16)
        prob = (counts / counts.sum()).astype(np.float16)
        g.records.append((encode_board(g.board), idx, prob,
                          g.board.turn == chess.WHITE))

        temp = 1.0 if g.plies < CFG.temp_moves else 0.0
        g.mcts.advance(g.mcts.best_move(temp))
        g.plies += 1

        outcome = g.board.outcome(claim_draw=True)
        if outcome is None and g.plies < CFG.max_game_plies:
            return False, 0

        if outcome is not None and outcome.winner is not None:
            z_white = 1 if outcome.winner == chess.WHITE else -1
            self.results["1-0" if z_white == 1 else "0-1"] += 1
        else:
            z_white = 0                      # draw, or adjudicated at max plies
            self.results["1/2-1/2"] += 1
        buffer.add_game(g.records, z_white)
        return True, len(g.records)
