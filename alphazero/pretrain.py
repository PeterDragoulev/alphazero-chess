"""
pretrain.py — supervised pretraining on human games streamed from Lichess.

Streams the Lichess/standard-chess-games HuggingFace dataset (billions of
games, no download), filtered to strong players, and trains the same net on
the same replay buffer and checkpoint as train.py:

  policy target = the move the human played (one-hot)
  value target  = the game result, side-to-move relative, blended with the
                  Stockfish evaluation where the game has one (--eval-weight)

Lichess analyses many games server-side and the evals ride along in the
movetext as [%eval] comments: ~9% of all games, but ~60% of 2400+ games.
Evals are converted to expected score with Lichess's own win-chance curve.
A game result is a noisy label for early positions (a won position thrown
away later reads as a loss); the eval is not, so blending gives the value
head a cleaner target. Positions without an eval keep the plain result.

This is ~1000x more positions/sec than self-play from a random net. The
ceiling is the strength of the training data; once progress flattens, switch
to self-play fine-tuning by simply running train.py — same data/, same
checkpoint, the buffer just refills with self-play games.

Usage:
    python pretrain.py                       # run until Ctrl+C (resumes)
    python pretrain.py --minutes 480         # overnight run
    python pretrain.py --min-elo 2200        # stronger but rarer games

Ingestion runs in --workers background processes (default 2): each streams
its own disjoint slice of the shuffled month-shards and parses/encodes games,
so the main process only fills the buffer and trains. --workers 0 runs
everything inline (single process, handy for debugging).

Optional local PGN source (--pgn FILE), e.g. an over-the-board master
database: games are filtered by rating (and to classical events), read in a
shuffled order, and mixed with the Lichess stream at --pgn-ratio (1.0 = PGN
only). Used for a final fine-tune; see finetune_otb.sh for the reversible
workflow.

Stop/resume semantics are identical to train.py (first Ctrl+C saves and
exits, second aborts). Each run streams in a freshly shuffled order, so
resumed sessions see new games rather than replaying the start of the stream.
Set HF_TOKEN in the environment for higher HuggingFace rate limits.
"""

import argparse
import csv
import io
import math
import multiprocessing as mp
import os
import queue
import random
import signal
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


def stream_games(min_elo: int, seed: int, stop, part: int = 0, nparts: int = 1):
    """
    Yields (movetext, z_white) for accepted games in a seed-shuffled shard
    order. With nparts > 1, yields only slice `part` of the shards, so
    parallel workers never see the same game. Survives network outages by
    rebuilding the stream with backoff; ends only on stop request.
    """
    from datasets import load_dataset

    while not stop.requested:
        try:
            shards = _shuffled_shards(seed)
            if shards is not None:
                shards = shards[part::nparts] or None
            ds = load_dataset(HF_REPO, data_files=shards, split="train",
                              streaming=True)
            for i, row in enumerate(ds):
                if stop.requested:
                    return
                if shards is None and i % nparts != part:
                    continue                 # listing failed: split by row
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


def encode_game(movetext: str):
    """
    Parse one game into compact arrays (planes (n,19,8,8) u8, move index (n,)
    u16, white-to-move (n,) bool), or None if unusable. Compact so it's cheap
    to send from a worker process.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(movetext))
    except Exception:
        return None
    return _encode_mainline(game)


def eval_to_value(pov_score) -> float:
    """White-relative engine eval -> expected score in (-1, 1), via Lichess's
    win-chance curve (fitted to real game outcomes): 1 pawn ~ +0.18,
    3 pawns ~ +0.51, mate = +-1."""
    s = pov_score.white()
    mate = s.mate()
    if mate is not None:
        return 1.0 if mate > 0 else -1.0
    return 2.0 / (1.0 + math.exp(-0.00368208 * s.score())) - 1.0


def _encode_mainline(game):
    """(planes, move index, white-to-move, white-relative eval or NaN)."""
    if game is None or game.errors:
        return None
    planes, idx, turns, evals = [], [], [], []
    board = game.board()
    prev_eval = math.nan                 # the eval comment follows the move,
    for node in game.mainline():         # so it scores the NEXT position
        move = node.move
        planes.append(encode_board(board))
        idx.append(move_to_index(move, board.turn))
        turns.append(board.turn == chess.WHITE)
        evals.append(prev_eval)
        board.push(move)
        pov = node.eval()
        prev_eval = eval_to_value(pov) if pov is not None else math.nan
    if len(planes) < MIN_PLIES:
        return None
    return (np.stack(planes), np.array(idx, dtype=np.uint16),
            np.array(turns, dtype=bool), np.array(evals, dtype=np.float32))


_ONE_HOT_PROB = np.array([1.0], dtype=np.float16)   # shared, never mutated


def add_encoded_game(buffer, game, z_white, eval_weight):
    """Add an encode_game() result to the buffer. Value target per position:
    (1 - w) * result + w * eval where an eval exists, else the result
    (all white-relative, flipped to side-to-move). Returns (positions,
    positions that had an eval)."""
    planes, idx, turns, evals = game
    has = ~np.isnan(evals)
    target = np.full(len(idx), float(z_white), dtype=np.float32)
    if eval_weight > 0:
        target[has] = (1 - eval_weight) * z_white + eval_weight * evals[has]
    for i in range(len(idx)):
        v = float(target[i])
        buffer.add(planes[i], idx[i:i + 1], _ONE_HOT_PROB, v if turns[i] else -v)
    return len(idx), int(has.sum())


# -- parallel ingestion ------------------------------------------------------------


# pyarrow/datasets streaming leaks ~1 GB/hour per process, so each worker is
# a small supervisor that runs the stream in a fresh child ("generation") and
# replaces it every WORKER_GENERATION_GAMES games or WORKER_GENERATION_SECONDS,
# whichever comes first — the leaked memory goes with the child. The time cap
# matters for strict filters: the leak tracks rows *read*, and at --min-elo
# 2400 a worker reads ~5x more rows per accepted game than at 2200.
WORKER_GENERATION_GAMES = 30_000
WORKER_GENERATION_SECONDS = 5 * 60


class _StreamStop:
    """Stop flag for a generation: its supervisor or the main process died."""

    def __init__(self, main_pid):
        self.main_pid = main_pid
        self.supervisor = os.getppid()

    @property
    def requested(self):
        if os.getppid() != self.supervisor:
            return True
        try:
            os.kill(self.main_pid, 0)
            return False
        except ProcessLookupError:
            return True


def _ingest_generation(out, min_elo, seed, part, nparts, main_pid):
    stop = _StreamStop(main_pid)
    sent = 0
    deadline = time.time() + WORKER_GENERATION_SECONDS
    try:
        for movetext, z in stream_games(min_elo, seed, stop, part, nparts):
            game = encode_game(movetext)
            if game is None:
                continue
            while True:                      # don't block forever if main is gone
                try:
                    out.put((game, z), timeout=1.0)
                    break
                except queue.Full:
                    if stop.requested:
                        return
            sent += 1
            if sent >= WORKER_GENERATION_GAMES or time.time() > deadline:
                break
        out.close()
        out.join_thread()                    # flush queued games before exiting
    finally:
        os._exit(0)                          # skip pyarrow's crashy finalization


def _ingest_worker(out, min_elo, seed, part, nparts, main_pid):
    """Supervisor: runs one streaming generation at a time until main exits."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # main handles Ctrl+C
    child = 0

    def _terminate(signum, frame):                   # main is shutting down
        if child:
            os.kill(child, signal.SIGKILL)
        os._exit(0)

    signal.signal(signal.SIGTERM, _terminate)
    generation = 0
    while True:
        try:
            os.kill(main_pid, 0)
        except ProcessLookupError:
            os._exit(0)
        # Same seed across workers within a generation, so their shard slices
        # stay disjoint; a new seed per generation reshuffles.
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            _ingest_generation(out, min_elo, seed + generation, part, nparts,
                               main_pid)
        os.waitpid(child, 0)
        child = 0
        generation += 1


def start_workers(n, min_elo, seed):
    """
    Fork n ingestion workers. Must run before CUDA is initialized: a forked
    child of a CUDA process is unusable (the workers never touch the GPU, but
    fork must still precede torch.cuda init to be safe).
    """
    ctx = mp.get_context("fork")
    out = ctx.Queue(maxsize=512)
    procs = [ctx.Process(target=_ingest_worker, daemon=True,
                         args=(out, min_elo, seed, i, n, os.getpid()))
             for i in range(n)]
    for p in procs:
        p.start()
    return out, procs


def worker_games(out, procs, stop):
    """Yields (encoded game, z_white) from the workers until stop."""
    while not stop.requested:
        try:
            yield out.get(timeout=1.0)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                raise RuntimeError("all ingestion workers died")


def inline_games(min_elo, seed, stop):
    """Single-process equivalent of worker_games (--workers 0)."""
    for movetext, z in stream_games(min_elo, seed, stop):
        game = encode_game(movetext)
        if game is not None:
            yield game, z


# -- optional local PGN source (e.g. over-the-board master games) -------------------

_FAST_EVENT_WORDS = (b"blitz", b"rapid", b"bullet", b"armageddon", b"speed",
                     b"online", b"titled tuesday")


def index_pgn(path, min_elo, classical_only):
    """
    Byte offsets of the games in `path` where both players are rated
    >= min_elo, with a decisive/drawn result, and (if classical_only) not
    from a rapid/blitz event. Scans headers only, so it's fast.
    """
    offsets, start, hdr = [], None, {}

    def keep():
        try:
            elo = min(int(hdr.get(b"WhiteElo", 0)), int(hdr.get(b"BlackElo", 0)))
        except ValueError:
            return False
        if elo < min_elo or hdr.get(b"Result") not in RESULT_Z_BYTES:
            return False
        event = hdr.get(b"Event", b"").lower()
        return not (classical_only and any(w in event for w in _FAST_EVENT_WORDS))

    with open(path, "rb") as f:
        pos = 0
        for line in f:
            if line.startswith(b"[Event "):
                if start is not None and keep():
                    offsets.append(start)
                start, hdr = pos, {}
            if line.startswith(b"[") and b' "' in line:
                key, _, val = line[1:].partition(b' "')
                hdr[key] = val.rsplit(b'"', 1)[0]
            pos += len(line)
    if start is not None and keep():
        offsets.append(start)
    return offsets


RESULT_Z_BYTES = {b"1-0", b"0-1", b"1/2-1/2"}


def _pgn_worker(out, path, offsets, seed, main_pid):
    """Reads the indexed games in a fresh shuffled order each pass, forever."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # main handles Ctrl+C
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    order = list(offsets)
    epoch = 0
    with open(path, encoding="latin-1") as f:
        while True:
            random.Random(seed + epoch).shuffle(order)
            for off in order:
                f.seek(off)
                try:
                    game = chess.pgn.read_game(f)
                except Exception:
                    continue
                arrays = _encode_mainline(game)
                if arrays is None:
                    continue
                item = (arrays, RESULT_Z[game.headers.get("Result", "*")])
                while True:
                    try:
                        out.put(item, timeout=1.0)
                        break
                    except queue.Full:
                        try:
                            os.kill(main_pid, 0)
                        except ProcessLookupError:
                            os._exit(0)
            epoch += 1
            print(f"(PGN source: finished pass {epoch} over {len(order):,} games)",
                  flush=True)


def start_pgn_worker(path, offsets, seed):
    """Fork the PGN reader. Like start_workers, must run before CUDA init."""
    ctx = mp.get_context("fork")
    out = ctx.Queue(maxsize=256)
    proc = ctx.Process(target=_pgn_worker, daemon=True,
                       args=(out, path, offsets, seed, os.getpid()))
    proc.start()
    return out, proc


def mixed_games(primary, secondary, ratio, rng):
    """Draw from `secondary` with probability `ratio`, else from `primary`."""
    while True:
        try:
            yield next(secondary) if rng.random() < ratio else next(primary)
        except StopIteration:            # a source ended (stop requested)
            return


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
    parser.add_argument("--workers", type=int, default=2,
                        help="ingestion processes (0 = parse inline)")
    parser.add_argument("--lr", type=float, default=None,
                        help="override the learning rate saved in the checkpoint")
    parser.add_argument("--pgn", default=None,
                        help="optional local PGN file to mix in (e.g. OTB master games)")
    parser.add_argument("--pgn-min-elo", type=int, default=2500,
                        help="both players must be at least this rating (PGN source)")
    parser.add_argument("--pgn-all-events", action="store_true",
                        help="keep rapid/blitz events from the PGN (default: classical only)")
    parser.add_argument("--pgn-ratio", type=float, default=0.5,
                        help="share of games taken from the PGN (1.0 = PGN only)")
    parser.add_argument("--eval-weight", type=float, default=0.5,
                        help="weight of the Stockfish eval in the value target where "
                             "a game has one (0 = game result only, the old behaviour)")
    parser.add_argument("--minutes", type=float, default=None,
                        help="stop gracefully after this many minutes")
    parser.add_argument("--blocks", type=int, default=None,
                        help="stop after this many blocks")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(CFG.data_dir, exist_ok=True)
    seed = int(time.time())                  # fresh shuffle every session
    procs = []
    pgn_only = args.pgn is not None and args.pgn_ratio >= 1.0
    if args.workers > 0 and not pgn_only:    # fork before CUDA init
        out, procs = start_workers(args.workers, args.min_elo, seed)
    if args.pgn is not None:
        t_idx = time.time()
        offsets = index_pgn(args.pgn, args.pgn_min_elo, not args.pgn_all_events)
        print(f"PGN {args.pgn}: {len(offsets):,} games with both players "
              f">= {args.pgn_min_elo}"
              f"{'' if args.pgn_all_events else ', classical events only'} "
              f"(indexed in {time.time() - t_idx:.0f}s)", flush=True)
        if not offsets:
            raise SystemExit("no PGN games pass the filter")
        pgn_out, pgn_proc = start_pgn_worker(args.pgn, offsets, seed)
        procs.append(pgn_proc)
    stop = GracefulStop()
    model, optimizer, scaler, games, steps = load_state(device)
    if args.lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        print(f"Learning rate set to {args.lr:g}")
    buffer = ReplayBuffer()
    print(f"Value target: game result"
          + (f" blended {args.eval_weight:g} with Stockfish evals where present"
             if args.eval_weight > 0 else " only"))

    if args.pgn is not None:
        pgn_stream = worker_games(pgn_out, [pgn_proc], stop)
    if pgn_only:
        stream = pgn_stream
        print("Training on the PGN source only...")
    else:
        stream_procs = [p for p in procs if p is not (pgn_proc if args.pgn else None)]
        stream = (worker_games(out, stream_procs, stop) if stream_procs
                  else inline_games(args.min_elo, seed, stop))
        print(f"Streaming Lichess games (min elo {args.min_elo}, seed {seed}, "
              f"{args.workers} workers)...")
        if args.pgn is not None:
            stream = mixed_games(stream, pgn_stream, args.pgn_ratio,
                                 random.Random(seed))
            print(f"Mixing in {args.pgn_ratio:.0%} PGN games")

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    block = 0
    try:
        while not stop.requested:
            t0 = time.time()
            positions = accepted = with_eval = 0
            while accepted < args.games_per_block and not stop.requested:
                try:
                    game, z_white = next(stream)
                except StopIteration:        # stream only ends on stop request
                    break
                n, n_eval = add_encoded_game(buffer, game, z_white, args.eval_weight)
                accepted += 1
                positions += n
                with_eval += n_eval

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
                  f"{pos_per_sec:,.0f} pos/s | evals {with_eval / max(positions, 1):.0%}",
                  flush=True)

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
    finally:
        for p in procs:
            p.terminate()

    print("Saved. Run pretrain.py again to continue, "
          "or train.py to fine-tune with self-play.")
    # pyarrow's streaming threads crash during interpreter finalization
    # (PyGILState_Release fatal error); everything is saved, so skip it.
    os._exit(0)


if __name__ == "__main__":
    main()
