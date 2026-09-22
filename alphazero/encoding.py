"""
encoding.py — board → tensor planes and move → policy index.

Everything is encoded from the side-to-move's perspective: when Black is to
move the board is mirrored vertically and colors are swapped. The network
therefore always sees "my pieces moving up the board", and its value output
is always "how good is this for the player to move". This halves what the
net has to learn.

Board planes (19, 8, 8), stored uint8:
  0-5   our   P N B R Q K
  6-11  their P N B R Q K
  12-13 our castling rights  (kingside, queenside)
  14-15 their castling rights
  16    en-passant square (one-hot)
  17    halfmove clock, raw 0-100 (divide by 100 when converting to float)
  18    all ones (lets convs detect the board edge)

Move encoding is AlphaZero's 73x64 scheme:
  planes 0-55   queen-style moves: 8 directions x 7 distances
  planes 56-63  knight moves
  planes 64-72  underpromotions: {N,B,R} x {capture-left, push, capture-right}
  (queen promotions are encoded as ordinary queen moves)
  index = plane * 64 + from_square   (squares mirrored when Black to move)
"""

import chess
import numpy as np

PLANES = 19
POLICY_SIZE = 73 * 64
HALFMOVE_PLANE = 17

_QUEEN_DIR_IDX = {(0, 1): 0, (1, 1): 1, (1, 0): 2, (1, -1): 3,
                  (0, -1): 4, (-1, -1): 5, (-1, 0): 6, (-1, 1): 7}
_KNIGHT_IDX = {(1, 2): 0, (2, 1): 1, (2, -1): 2, (1, -2): 3,
               (-1, -2): 4, (-2, -1): 5, (-2, 1): 6, (-1, 2): 7}
_UNDERPROMO_IDX = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING)


def encode_board(board: chess.Board) -> np.ndarray:
    """Return (19, 8, 8) uint8 planes from the side-to-move's perspective."""
    planes = np.zeros((PLANES, 8, 8), dtype=np.uint8)
    us = board.turn
    them = not us
    flip = us == chess.BLACK

    # Piece planes straight from the 12 bitboards: bit i of a bitboard is
    # square i = rank*8 + file, so little-endian unpacking lands each bit at
    # [rank, file]. Mirroring for Black is a byte swap (chess.flip_vertical).
    masks = [board.pieces_mask(pt, color)
             for color in (us, them) for pt in _PIECE_TYPES]
    if flip:
        masks = [chess.flip_vertical(m) for m in masks]
    raw = b"".join(m.to_bytes(8, "little") for m in masks)
    planes[:12] = np.unpackbits(np.frombuffer(raw, dtype=np.uint8),
                                bitorder="little").reshape(12, 8, 8)

    if board.has_kingside_castling_rights(us):
        planes[12] = 1
    if board.has_queenside_castling_rights(us):
        planes[13] = 1
    if board.has_kingside_castling_rights(them):
        planes[14] = 1
    if board.has_queenside_castling_rights(them):
        planes[15] = 1

    if board.ep_square is not None:
        ep = chess.square_mirror(board.ep_square) if flip else board.ep_square
        rank, file = divmod(ep, 8)
        planes[16, rank, file] = 1

    planes[HALFMOVE_PLANE] = min(board.halfmove_clock, 100)
    planes[18] = 1
    return planes


def planes_to_float(u8: np.ndarray) -> np.ndarray:
    """Convert stored uint8 planes (any leading batch dims) to network input."""
    f = u8.astype(np.float32)
    f[..., HALFMOVE_PLANE, :, :] /= 100.0
    return f


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Map a legal move to its policy index, given whose turn it is."""
    frm, to = move.from_square, move.to_square
    if turn == chess.BLACK:
        frm = chess.square_mirror(frm)
        to = chess.square_mirror(to)
    df = (to & 7) - (frm & 7)
    dr = (to >> 3) - (frm >> 3)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = 64 + _UNDERPROMO_IDX[move.promotion] * 3 + (df + 1)
    elif (df, dr) in _KNIGHT_IDX:
        plane = 56 + _KNIGHT_IDX[(df, dr)]
    else:
        dist = max(abs(df), abs(dr))
        direction = ((df > 0) - (df < 0), (dr > 0) - (dr < 0))
        plane = _QUEEN_DIR_IDX[direction] * 7 + (dist - 1)
    return plane * 64 + frm
