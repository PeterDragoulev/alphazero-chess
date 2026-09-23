"""
config.py — all hyperparameters in one place.

Defaults are tuned for an RTX 3070 Ti laptop (8 GB VRAM): a ~3.5M-param net,
self-play inference batched across 64 parallel games, fp16 everywhere.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- network ---
    input_planes: int = 19
    channels: int = 128
    blocks: int = 10
    policy_size: int = 73 * 64          # AlphaZero move encoding: 4672

    # --- MCTS ---
    c_puct: float = 1.5
    fpu_reduction: float = 0.25         # unvisited move Q = parent Q - this*sqrt(explored prior)
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # --- self-play ---
    simulations: int = 128              # MCTS sims per move during self-play
    parallel_games: int = 64            # games played in lockstep (= GPU batch)
    temp_moves: int = 20                # sample moves ∝ visits for first N plies
    max_game_plies: int = 300           # adjudicate as draw beyond this

    # --- training ---
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    sample_ratio: float = 2.0           # each new position trained on ~2x
    min_buffer: int = 8_192             # don't train before this many positions
    buffer_capacity: int = 500_000      # ~600 MB RAM as uint8 planes

    # --- orchestration ---
    games_per_block: int = 32           # checkpoint after this many finished games

    # --- play (engine.py / evaluate.py) ---
    play_simulations: int = 6400          # new sims per move (batched, tree reused: ~5 s)
    play_batch: int = 16                # leaves per GPU call when playing (virtual loss)

    # --- paths ---
    data_dir: str = "data"
    checkpoint: str = "data/checkpoint.pt"
    pretrained_weights: str = "weights/lichess_5.66M_128x10.pt"   # shipped net (June net: pretrained_128x10.pt)
    buffer_dir: str = "data/buffer"
    book_path: str = "books/komodo.bin"  # polyglot; optional (engine skips if absent)


CFG = Config()
