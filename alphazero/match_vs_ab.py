"""Quick match: active neural net (data/checkpoint.pt) vs alpha-beta depth-4."""
import math
import sys
import time

import chess

from engine import choose_move, _material_move, MODEL_PATH, _model
from ui import AB_DEPTH, NN_SIMS, MAX_AUTO_PLIES

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else AB_DEPTH


def play_game(nn_color: bool) -> str:
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) \
            and len(board.move_stack) < MAX_AUTO_PLIES:
        if board.turn == nn_color:
            mv = choose_move(board, simulations=NN_SIMS)
        else:
            mv = _material_move(board, depth=DEPTH)
        if mv is None:
            break
        board.push(mv)
    return board.result(claim_draw=True), board.outcome(claim_draw=True)


def main():
    print(f"Neural net ({MODEL_PATH}, loaded={_model is not None}) "
          f"vs alpha-beta d{DEPTH}, {GAMES} games, NN {NN_SIMS} sims/move",
          flush=True)
    w = d = l = 0
    t0 = time.time()
    for i in range(GAMES):
        nn_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        _, outcome = play_game(nn_color)
        if outcome is None or outcome.winner is None:
            d += 1
            r = "draw"
        elif outcome.winner == nn_color:
            w += 1
            r = "NN win"
        else:
            l += 1
            r = "NN loss"
        print(f"game {i+1}/{GAMES} (NN={'W' if nn_color else 'B'}): {r:8s} "
              f"running +{w} ={d} -{l}  [{time.time()-t0:.0f}s]", flush=True)

    total = w + d + l
    score = (w + 0.5 * d) / total
    sc = min(max(score, 1 / (2 * total)), 1 - 1 / (2 * total))
    elo = 400 * math.log10(sc / (1 - sc))
    print(f"\nRESULT  neural net vs alpha-beta d{DEPTH}:  "
          f"+{w} ={d} -{l}  score {score:.2f}  Elo {elo:+.0f}", flush=True)


if __name__ == "__main__":
    main()
