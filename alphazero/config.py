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
    se: bool = False                    # squeeze-and-excitation in each block (fresh nets; checkpoints record their own)
    policy_size: int = 73 * 64          # AlphaZero move encoding: 4672

    # --- MCTS ---
    c_puct: float = 1.5
    fpu_reduction: float = 0.25         # unvisited move Q = parent Q - this*sqrt(explored prior)
    native_mcts: bool = True            # C++ tree (native/build.sh) when built; False = mcts.PyMCTS
    policy_temp: float = 1.3            # native only: priors = softmax(logits / T); 1.3 tuned (+38 Elo @3200 nodes)
    cpuct_base: float = 38739.0         # native only: c_puct grows as c + factor*ln((N+base)/base)
    cpuct_factor: float = 0.0           #   (0 = constant c_puct, the original rule)
    contempt: float = 0.0               # native only: draws = -contempt for us when root Q > threshold
    contempt_threshold: float = 0.1
    q_select: float = 0.5               # native: final move = best Q among moves with >= this share of max visits (0 = most visits); +69 @800, +19 @3200
    eval_cache: int = 500_000           # native: cached network outputs for transpositions (entries; 0 = off); 21-38% of leaves hit
    solver: bool = True                 # native: MCTS-solver (proven mates propagate up the tree); tested neutral, proves mates
    smart_time: bool = False            # UCI: stop early if the best move can't be caught; extend (<=1.5x) if unstable
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # --- self-play ---
    simulations: int = 128              # MCTS sims per move during self-play
    parallel_games: int = 64            # games played in lockstep (= GPU batch)
    temp_moves: int = 20                # sample moves ∝ visits for first N plies
    max_game_plies: int = 300           # adjudicate as draw beyond this
    playout_cap_prob: float = 1.0       # <1: that share of moves get a full search (recorded)
    fast_simulations: int = 32          # search size for the other moves (playout cap)

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
    play_batch: int = 32                # leaves per GPU call when playing (virtual loss); 32 tuned: +115 Elo vs 16 at equal time
    play_pipeline: bool = False         # native: keep 2 batches in flight (CPU selects while GPU evaluates)

    # --- paths ---
    data_dir: str = "data"
    checkpoint: str = "data/checkpoint.pt"
    pretrained_weights: str = "weights/lichess_otb_7.12M_128x10.pt"   # shipped net (older: lichess_5.66M, pretrained_128x10)
    buffer_dir: str = "data/buffer"
    book_path: str = "books/komodo.bin"  # polyglot; optional (engine skips if absent)


CFG = Config()
