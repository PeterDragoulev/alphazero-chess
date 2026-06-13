"""
pretrain.py — supervised pretraining on human games streamed from Lichess.

Streams the Lichess/standard-chess-games HuggingFace dataset (billions of
games, no download), filtered to strong players, and trains the same net on
the same replay buffer and checkpoint as train.py:

  policy target = the move the human played (one-hot)
  value target  = the game result, side-to-move relative

This is ~1000x more positions/sec than self-play from a random net. The
ceiling is the strength of the training data; once progress flattens, switch
to self-play fine-tuning by simply running train.py — same data/, same
checkpoint, the buffer just refills with self-play games.

Usage:
    python pretrain.py                       # run until Ctrl+C (resumes)
    python pretrain.py --minutes 480         # overnight run
    python pretrain.py --min-elo 2200        # stronger but rarer games

Stop/resume semantics are identical to train.py (first Ctrl+C saves and
exits, second aborts). Each run streams in a freshly shuffled order, so
resumed sessions see new games rather than replaying the start of the stream.
Set HF_TOKEN in the environment for higher HuggingFace rate limits.
"""

import argparse
import csv
import io
import os
import random
import time

import chess
import chess.pgn
import numpy as np
import torch

from config import CFG
from encoding import encode_board, move_to_index
from replay_buffer import ReplayBuffer
from train import GracefulStop, load_state, save_state, train_steps

LOG_PATH = os.path.join(CFG.data_dir, "pretrain_log.csv")
RESULT_Z = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
MIN_PLIES = 10                  # skip aborted / instantly-decided games
MIN_BASE_SECONDS = 180          # skip bullet: too low quality
HF_REPO = "Lichess/standard-chess-games"


def _shuffled_shards(seed: int):
    """
    Per-month parquet shard paths for the dataset, in a seed-shuffled order.

    This replaces datasets' streaming `.shuffle(buffer_size=...)`, which on this
    dataset balloons to ~10 GB RSS (each buffered row pins a decompressed
    pyarrow row-group) regardless of buffer size — it OOM-kills an 11 GB box.
    Shuffling the ~26k month-shards and reading them sequentially instead keeps
    RSS under ~1 GB, still gives a different (and modern, not 2013-first) game
    mix every session, and streams at several thousand games/sec. Returns None
    if the listing fails, so the caller can fall back to plain streaming.
    """
    try:
        from huggingface_hub import HfFileSystem
        prefix = f"datasets/{HF_REPO}/"
        files = [f[len(prefix):]
                 for f in HfFileSystem().glob(f"{prefix}**/*.parquet")]
        random.Random(seed).shuffle(files)
        return files or None
    except Exception as e:
        print(f"Shard listing failed ({type(e).__name__}: {e}); "
              f"falling back to sequential streaming.", flush=True)
        return None


def stream_games(min_elo: int, seed: int, stop):
    """
    Yields (movetext, z_white) for accepted games in a seed-shuffled shard
    order. Survives network outages by rebuilding the stream with backoff;
    ends only on stop request.
    """
    from datasets import load_dataset

    while not stop.requested:
        try:
            shards = _shuffled_shards(seed)
            ds = load_dataset(HF_REPO, data_files=shards, split="train",
                              streaming=True)
            for row in ds:
                if stop.requested:
                    return
                z = RESULT_Z.get(row["Result"])
                if z is None:
                    continue
                if not row["WhiteElo"] or not row["BlackElo"]:
                    continue
                if min(row["WhiteElo"], row["BlackElo"]) < min_elo:
                    continue
                if row["Termination"] not in ("Normal", "Time forfeit"):
                    continue
                tc = row["TimeControl"]
                if tc and tc != "-":         # "-" = correspondence: keep
                    try:
                        if int(tc.split("+")[0]) < MIN_BASE_SECONDS:
                            continue
                    except ValueError:
                        continue
                yield row["movetext"], z
            seed += 1                        # dataset exhausted: reshuffle
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except Exception as e:
            print(f"Stream error ({type(e).__name__}: {e}) — "
                  f"reconnecting in 30s...", flush=True)
            seed += 1
            for _ in range(30):
                if stop.requested:
                    return
                time.sleep(1)


def game_records(movetext: str):
    """Parse one game into replay-buffer records, or None if unusable."""
    try:
        game = chess.pgn.read_game(io.StringIO(movetext))
    except Exception:
        return None
    if game is None or game.errors:
        return None
    records = []
    board = game.board()
    for move in game.mainline_moves():
        records.append((
            encode_board(board),
            np.array([move_to_index(move, board.turn)], dtype=np.uint16),
            np.array([1.0], dtype=np.float16),
            board.turn == chess.WHITE,
        ))
        board.push(move)
    return records if len(records) >= MIN_PLIES else None


def log_block(games, steps, buffer_size, p_loss, v_loss, pos_per_sec):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "games", "steps", "buffer",
                        "policy_loss", "value_loss", "positions_per_sec"])
        w.writerow([int(time.time()), games, steps, buffer_size,
                    f"{p_loss:.4f}", f"{v_loss:.4f}", f"{pos_per_sec:.0f}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-elo", type=int, default=2000,
                        help="both players must be at least this rating")
    parser.add_argument("--games-per-block", type=int, default=1000,
                        help="games ingested between checkpoints")
    parser.add_argument("--sample-ratio", type=float, default=1.0,
                        help="train steps per new position (1.0 = each ~once)")
    parser.add_argument("--minutes", type=float, default=None,
                        help="stop gracefully after this many minutes")
    parser.add_argument("--blocks", type=int, default=None,
                        help="stop after this many blocks")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(CFG.data_dir, exist_ok=True)
    stop = GracefulStop()
    model, optimizer, scaler, games, steps = load_state(device)
    buffer = ReplayBuffer()

    seed = int(time.time())                  # fresh shuffle every session
    stream = stream_games(args.min_elo, seed, stop)
    print(f"Streaming Lichess games (min elo {args.min_elo}, seed {seed})...")

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    block = 0
    try:
        while not stop.requested:
            t0 = time.time()
            positions = accepted = 0
            while accepted < args.games_per_block and not stop.requested:
                try:
                    movetext, z_white = next(stream)
                except StopIteration:        # stream only ends on stop request
                    break
                records = game_records(movetext)
                if records is None:
                    continue
                buffer.add_game(records, z_white)
                accepted += 1
                positions += len(records)

            p_loss = v_loss = float("nan")
            if buffer.size >= CFG.min_buffer and positions > 0:
                n_steps = max(1, round(positions * args.sample_ratio
                                       / CFG.batch_size))
                p_loss, v_loss = train_steps(model, optimizer, scaler,
                                             buffer, n_steps, device)
                steps += n_steps
            games += accepted

            buffer.save()
            save_state(model, optimizer, scaler, games, steps)
            pos_per_sec = positions / max(time.time() - t0, 1e-9)
            log_block(games, steps, buffer.size, p_loss, v_loss, pos_per_sec)
            print(f"[block {block:4d}] games {games:,} | buffer {buffer.size:,} | "
                  f"steps {steps:,} | loss p {p_loss:.3f} v {v_loss:.3f} | "
                  f"{pos_per_sec:,.0f} pos/s", flush=True)

            block += 1
            if args.blocks is not None and block >= args.blocks:
                break
            if deadline is not None and time.time() >= deadline:
                print("Time limit reached.")
                break
    except Exception:
        # unexpected crash: salvage what we have, then surface the error
        buffer.save()
        save_state(model, optimizer, scaler, games, steps)
        raise

    print("Saved. Run pretrain.py again to continue, "
          "or train.py to fine-tune with self-play.")
    # pyarrow's streaming threads crash during interpreter finalization
    # (PyGILState_Release fatal error); everything is saved, so skip it.
    os._exit(0)


if __name__ == "__main__":
    main()
