"""
model.py — CNN board evaluator.

Input:  (batch, 18, 8, 8)  float32 tensor
  Planes 0-5:   white  P N B R Q K  (1 where piece present)
  Planes 6-11:  black  P N B R Q K
  Plane  12:    side to move (all-1 = white, all-0 = black)
  Planes 13-16: castling rights (WK WQ BK BQ)
  Plane  17:    en-passant file (column filled with 1 if ep available)

Output: scalar in (-1, 1) from the perspective of WHITE.
"""

import chess
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Board → tensor
# ---------------------------------------------------------------------------

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Return a (18, 8, 8) float32 tensor for the given board."""
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        rank, file = divmod(sq, 8)
        offset = PIECE_ORDER.index(piece.piece_type)
        if piece.color == chess.WHITE:
            planes[offset, rank, file] = 1.0
        else:
            planes[6 + offset, rank, file] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[12] = 1.0

    # Castling
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16] = 1.0

    # En-passant
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        planes[17, :, ep_file] = 1.0

    return torch.from_numpy(planes)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class BoardEvaluator(nn.Module):
    """
    Lightweight residual CNN.  ~500 k parameters — fast enough for alpha-beta
    call-outs while still learning positional patterns.
    """
    def __init__(self, channels: int = 64, num_res: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res)]
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),          # output in (-1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_tower(x)
        return self.value_head(x).squeeze(-1)   # (batch,)
