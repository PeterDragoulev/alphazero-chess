"""
evaluate.py — arena: pit two checkpoints against each other.

    python evaluate.py data/checkpoint.pt data/old.pt --games 20 --sims 200

Colors alternate every game; the first few plies are sampled at temperature 1
so games differ (otherwise deterministic play repeats one game). Prints
W/D/L for checkpoint A and a rough Elo difference.

Tip: before a long run, copy the current checkpoint
(cp data/checkpoint.pt data/old.pt) so you have a baseline to compare with.
"""

import argparse
import math

import chess
import torch

from mcts import MCTS
from network import Evaluator, load_checkpoint

OPENING_TEMP_PLIES = 8
MAX_PLIES = 300


def play_game(ev_a: Evaluator, ev_b: Evaluator, a_is_white: bool,
              sims: int) -> float:
    """Returns score for A: 1 win, 0.5 draw, 0 loss."""
    # Each tree owns its own board; advance() keeps them in sync.
    mcts_w = MCTS(chess.Board())
    mcts_b = MCTS(chess.Board())
    board = mcts_w.board                      # ground truth for outcome checks

    while len(board.move_stack) < MAX_PLIES:
        if board.outcome(claim_draw=True) is not None:
            break
        white_to_move = board.turn == chess.WHITE
        mcts = mcts_w if white_to_move else mcts_b
        ev = ev_a if white_to_move == a_is_white else ev_b
        for _ in range(sims):
            planes = mcts.select_leaf()
            if planes is not None:
                logits, values = ev(planes[None])
                mcts.expand_backup(logits[0], float(values[0]))
        temp = 1.0 if len(board.move_stack) < OPENING_TEMP_PLIES else 0.0
        move = mcts.best_move(temp)
        mcts_w.advance(move)
        mcts_b.advance(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if (outcome.winner == chess.WHITE) == a_is_white else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_a")
    parser.add_argument("ckpt_b")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, _ = load_checkpoint(args.ckpt_a, device)
    model_b, _ = load_checkpoint(args.ckpt_b, device)
    model_a.eval(), model_b.eval()
    ev_a, ev_b = Evaluator(model_a, device), Evaluator(model_b, device)

    wins = draws = losses = 0
    for i in range(args.games):
        score = play_game(ev_a, ev_b, a_is_white=(i % 2 == 0), sims=args.sims)
        wins += score == 1.0
        draws += score == 0.5
        losses += score == 0.0
        print(f"game {i + 1}/{args.games}: "
              f"{'win' if score == 1 else 'draw' if score == 0.5 else 'loss'} "
              f"for A (running {wins}/{draws}/{losses})")

    total = wins + draws + losses
    rate = (wins + 0.5 * draws) / total
    rate_c = min(max(rate, 1 / (2 * total)), 1 - 1 / (2 * total))
    elo = 400 * math.log10(rate_c / (1 - rate_c))
    print(f"\n{args.ckpt_a} vs {args.ckpt_b}: "
          f"+{wins} ={draws} -{losses}  score {rate:.2f}  Elo {elo:+.0f}")


if __name__ == "__main__":
    main()
