"""
benchmark.py — same structure as before; engine vs random.
"""

import random
import chess

from engine import choose_move

GAMES = 20
MAX_PLIES = 160


def random_move(board: chess.Board) -> chess.Move:
    return random.choice(list(board.legal_moves))


def play_game(engine_as_white: bool) -> str:
    board = chess.Board()
    for _ in range(MAX_PLIES):
        if board.is_game_over():
            break
        if (board.turn == chess.WHITE) == engine_as_white:
            move = choose_move(board)
        else:
            move = random_move(board)
        if move is None:
            break
        board.push(move)
    return board.result()


def estimate_elo(win_rate: float) -> int:
    return int(600 + win_rate * 600)


def main() -> None:
    wins = losses = draws = 0

    for i in range(GAMES):
        engine_as_white = i % 2 == 0
        result = play_game(engine_as_white)
        if result == "1-0":
            if engine_as_white:
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            if engine_as_white:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1
        print(f"Game {i + 1}/{GAMES}: {result}")

    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total if total else 0.0
    elo = estimate_elo(win_rate)

    print("\nBenchmark vs random:")
    print(f"Wins: {wins}  Losses: {losses}  Draws: {draws}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Rough ELO estimate: {elo}")


if __name__ == "__main__":
    main()
