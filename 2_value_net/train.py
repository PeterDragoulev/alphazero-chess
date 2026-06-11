"""
train.py — train BoardEvaluator on the angeluriot/chess_games HuggingFace dataset.

Strategy (imitation / supervised):
  For every position in a game, the target value is the game outcome
  from White's perspective:  +1 = White won, -1 = Black won, 0 = draw.
  The network learns to predict that outcome from the board state.

Usage:
    python train.py                        # defaults
    python train.py --games 500000 --epochs 3 --lr 1e-3
"""

import argparse
import os
import time

import chess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from model import BoardEvaluator, board_to_tensor

MODEL_PATH = "evaluator.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


def _iter_positions(max_games: int, skip_games: int = 0):
    """
    Yield (tensor, target) pairs by streaming the HuggingFace dataset.
    Skips the first `skip_games` games (for resuming across epochs).
    """
    from datasets import load_dataset

    ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)
    count = 0

    for game in ds:
        if count < skip_games:
            count += 1
            continue
        if count >= skip_games + max_games:
            break

        result_str = game.get("winner")
        # Map winner field: 'white' / 'black' / None (draw)
        if result_str == "white":
            target = 1.0
        elif result_str == "black":
            target = -1.0
        else:
            target = 0.0

        moves_uci = game.get("moves_uci", [])
        if not moves_uci:
            count += 1
            continue

        board = chess.Board()
        for uci in moves_uci:
            try:
                board.push_uci(uci)
            except Exception:
                break
            yield board_to_tensor(board), torch.tensor(target, dtype=torch.float32)

        count += 1


class StreamingChessDataset(IterableDataset):
    def __init__(self, max_games: int, skip_games: int = 0):
        self.max_games = max_games
        self.skip_games = skip_games

    def __iter__(self):
        return _iter_positions(self.max_games, self.skip_games)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load or create model
    model = BoardEvaluator(channels=args.channels, num_res=args.res_blocks).to(device)
    start_epoch = 0
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from checkpoint (epoch {start_epoch})")
    else:
        print("Starting fresh model")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\n=== Epoch {epoch + 1} | games: {args.games} ===")
        dataset = StreamingChessDataset(max_games=args.games)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

        model.train()
        running_loss = 0.0
        batches = 0
        t0 = time.time()

        for tensors, targets in loader:
            tensors = tensors.to(device)
            targets = targets.to(device)

            preds = model(tensors)
            loss = criterion(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            batches += 1

            if batches % 500 == 0:
                avg = running_loss / batches
                elapsed = time.time() - t0
                print(f"  batch {batches:6d} | avg loss {avg:.4f} | {elapsed:.0f}s elapsed")

        scheduler.step()
        avg_loss = running_loss / max(batches, 1)
        print(f"Epoch {epoch + 1} done | avg loss {avg_loss:.4f} | batches {batches}")

        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, MODEL_PATH)
        print(f"Saved → {MODEL_PATH}")

    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games",      type=int,   default=100_000,
                        help="Games per epoch (streamed, no download needed)")
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--channels",   type=int,   default=64,
                        help="CNN channel width (64 = ~500k params)")
    parser.add_argument("--res-blocks", type=int,   default=6)
    args = parser.parse_args()
    train(args)
