# Chess Engine with AlphaZero-Style Self-Play

A chess engine built up in stages, from classical game-tree search to an
AlphaZero-style policy-value network guided by Monte-Carlo Tree Search, and
trained entirely on one laptop GPU (RTX 3070 Ti, 8 GB VRAM).

| Stage | Folder | Idea |
|---|---|---|
| 1 | [`1_search/`](1_search) | Minimax, then alpha-beta pruning over a material evaluator |
| 2 | [`2_value_net/`](2_value_net) | Replace the hand-written evaluator with a supervised CNN value network |
| 3 | [`alphazero/`](alphazero) | Policy + value ResNet, PUCT MCTS, supervised pretraining on Lichess, then self-play |
| — | [`benchmarks/`](benchmarks) | Reproducible MCTS speedup benchmark (batched leaf evaluation, push/pop traversal) |

**Headline numbers** (all measured on this repo; see [Results](#results)):

- Alpha-beta finds the same move scores as minimax while searching **7–52x fewer nodes**.
- Self-play MCTS runs **~16–18x faster** than the naive implementation:
  ~14–17x from batching leaf evaluations across 64 games into single GPU calls,
  plus **~10–15%** from push/pop board traversal.
- The pretrained network (1.78M strong Lichess games, ~8 GPU-hours) **beat a
  depth-4 alpha-beta searcher 5–0 and a depth-5 searcher 1–0**.
- First-play urgency (FPU) in the search: **+102 Elo** (95% CI +66..+140) in a
  200-game paired arena, same net and simulation budget.
- Play-time search is **4x faster** per simulation (virtual-loss batching within
  one tree, CUDA graphs, tree reuse), which in the same ~1.2 s per move takes
  the principal variation from **~12 to ~17 plies**.
- Pretraining ingestion went from ~4,300 to **~6,900 positions/s** with
  parallel worker processes and a 1.9x faster board encoder.

---

## Stage 1 — Minimax and alpha-beta ([`1_search/search.py`](1_search/search.py))

Chess is a two-player zero-sum game, so the best move is the one that
maximizes your score assuming the opponent then minimizes it, recursively.
Both searches are written in **negamax** form: a score is always from the
side to move's point of view, so `score(parent) = max(-score(child))` and one
function handles both colors.

- **Evaluator:** material count in centipawns (P=100, N=320, B=330, R=500, Q=900).
  Checkmate scores ±100,000, adjusted by ply so the engine prefers the fastest mate.
- **Minimax** visits every node to a fixed depth. The branching factor is ~35,
  so it grows as 35^d.
- **Alpha-beta** returns the same value but stops searching a move once it's
  proven worse than something already found (`score >= beta`: the opponent
  would never allow this line). How much it prunes depends on move order, so
  captures are tried first, ordered by **MVV-LVA** (most valuable victim,
  least valuable attacker).
- Both use **make/unmake traversal** (`board.push` / `board.pop`) on a single
  board rather than copying the position at every node.

```
$ python 1_search/search.py          # depth 3
position         algorithm  move     score     nodes
hanging queen    minimax    Bxg5       900    50,587
hanging queen    alphabeta  Bxg5       900       967    -> 52x fewer nodes
back-rank mate   minimax    Ra8#     99999     3,201
back-rank mate   alphabeta  Ra8#     99999       472    ->  7x fewer nodes
middlegame       minimax    Qa4        200    39,074
middlegame       alphabeta  cxd5       200     1,375    -> 28x fewer nodes
```

(In the middlegame, the two pick different moves that tie on score. Only the
score is guaranteed to match.)

**Limitation that motivated stage 2:** a material-only evaluator knows nothing
about king safety, pawn structure or piece activity. It also suffers from the
*horizon effect*: anything that happens one ply past the search depth is
invisible.

## Stage 2 — Supervised value network + alpha-beta ([`2_value_net/`](2_value_net))

The material count is replaced by a learned evaluator:

- **Model** (`model.py`): a residual CNN (64 channels × 6 blocks, ~500k
  parameters). The input is 18 planes of 8×8 (12 piece planes, side to move,
  castling rights, en-passant file). The output is `tanh` ∈ (−1, 1).
- **Training** (`train.py`): supervised regression. Every position of every
  game streamed from the `angeluriot/chess_games` HuggingFace dataset is
  labeled with that game's final result (+1 / 0 / −1), and the net is trained
  with MSE loss.
- **Engine** (`engine.py`): the same alpha-beta with MVV-LVA ordering as
  stage 1, plus an optional polyglot opening book. The net scores the leaves.

**Why this wasn't enough:** a network call per leaf is expensive, so the search
could only afford depth 3. The net also only says *how good* a position is, not
*which moves are worth looking at*, so alpha-beta still has to consider every
legal move. AlphaZero solves both problems with a **policy head**, which tells
the search where to look, and MCTS, which spends its simulations on the
promising lines.

## Stage 3 — AlphaZero-style engine ([`alphazero/`](alphazero))

```
pretrain.py     supervised: streams Lichess games, policy = move played, value = result
train.py        self-play loop: self-play block → gradient steps → checkpoint, repeat
selfplay.py     64 games in lockstep; all their MCTS leaves evaluated in ONE GPU batch
mcts.py         PUCT search with externalized (batchable) evaluation, push/pop traversal
network.py      policy + value ResNet (3.66M params), fp16 batched Evaluator
encoding.py     board → (19,8,8) uint8 planes; move → AlphaZero 73×64 policy index
replay_buffer.py disk-persisted 500k-position ring buffer, sparse policy targets
engine.py       play interface: opening book → endgame tablebase → MCTS
evaluate.py     checkpoint-vs-checkpoint arena with Elo estimate
config.py       every hyperparameter in one place
```

### Board and move encoding (`encoding.py`)

- **Everything is relative to the side to move.** When Black is to move, the
  board is mirrored and the colors swapped, so the network always sees "my
  pieces moving up the board". The value head likewise always means "how good
  is this for the player to move". This halves what the net has to learn.
- **19 input planes:** 6 planes for our pieces, 6 for theirs, 4 castling
  rights, en-passant square, halfmove clock, and an all-ones plane that lets
  convolutions detect the board edge. They are stored as `uint8` (~1.2 KB per
  position) and converted to float only per batch.
- **4,672-way policy (AlphaZero's 73×64 scheme):** for each from-square,
  56 queen-style moves (8 directions × 7 distances), 8 knight moves and
  9 underpromotions. Every legal chess move maps to exactly one index.

### Network (`network.py`)

A ResNet with a 128-channel stem and 10 residual blocks (**3.66M
parameters**), sized to fit the 8 GB VRAM budget. It has two heads:

- **Policy:** 3×3 conv → 1×1 conv to 73 planes → 4,672 logits
- **Value:** 1×1 conv → FC 256 → `tanh` scalar

The loss is `cross_entropy(policy, π) + MSE(value, z)`, trained with AdamW
under fp16 autocast and a GradScaler, with gradient clipping at 1.0.
Inference goes through `Evaluator`, which takes uint8 planes in and returns
numpy arrays out, running in fp16.

### MCTS (`mcts.py`)

The search is PUCT, as in AlphaZero. At each node it picks the move maximizing

```
Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))
```

where `P` is the network's prior, `Q` the mean backed-up value and `N` the
visit count. Statistics live on edges as numpy arrays, so the selection is
vectorized. Dirichlet noise (α=0.3, ε=0.25) at the root keeps self-play
exploring. The subtree under the chosen move is reused for the next move.

**The core trick is externalized evaluation.** The tree never calls the
network itself. A simulation is split in two:

```python
planes = mcts.select_leaf()       # descend to an unexpanded leaf, return its planes
...                               # caller batches leaves from MANY trees
mcts.expand_backup(logits, value) # expand the leaf with the priors, back up the value
```

This lets `selfplay.py` step 64 games in lockstep. Every simulation round
collects one leaf from each game and evaluates all 64 **in a single GPU call**.

**Push/pop traversal:** each tree owns a private search board. A simulation
pushes moves on the way down and pops back to the root after the leaf is
handled, instead of copying the board every simulation. Legal moves are
generated once per leaf and reused for both terminal detection (no legal moves
means checkmate or stalemate) and expansion. Previously they were generated up
to three times per simulation (`is_checkmate`, `is_stalemate`, expansion).

**First-play urgency (FPU).** Plain AlphaZero scores a move that hasn't been
visited yet as Q = 0, i.e. "probably a draw". In a losing position every
untried move then looks better than the tried ones, so the search sprays its
visits across junk. Following Lc0, an unvisited move is instead scored at the
parent's current average minus `0.25 · √(prior mass already explored)`. In a
200-game paired arena (same net, 200 simulations, each opening played with
both colors) this scored **+83 =91 −26, +102 Elo (95% CI +66..+140)** against
the Q = 0 rule.

**Batching within one tree (virtual loss).** Self-play batches across 64
games, but a single game against a human only has one tree. `select_leaves(k)`
descends k times; each descent adds a *virtual loss* to the edges it walks, as
if that line had just lost, so the next descent picks a different line. The k
leaves go to the GPU in one call, then `expand_leaves()` removes the virtual
losses and backs up the real values. Virtual losses live in a separate integer
array, so removing them is exact: a batch of 1 is bit-identical to the
single-leaf search.

**Tree reuse and repetitions.** When playing, the engine keeps its tree
between moves and continues from the subtree under the moves actually played.
A leaf that repeats a position from the game, or an earlier position on its
own search line, is scored as a draw (the usual in-search twofold rule). In
tests, the winning side stopped playing the repeating move (the old search
scored it +0.88; the new one scores it 0.00), and the losing side took the
repetition instead of playing on at −0.67.

For benchmarking, `MCTS(..., fpu_reduction=None, repetition_draws=False)`
switches both of these off and reproduces the original search exactly.

### Training pipeline

**1. Supervised pretraining (`pretrain.py`).** Self-play starting from a
random network produces ~6 positions/sec on this GPU. Streaming human games
produces ~5,000. So the net is first trained on strong human games:

- Games are streamed from the `Lichess/standard-chess-games` HuggingFace
  dataset, with no full download. Filters: both players rated ≥ 2000, no
  bullet, normal termination, at least 10 plies.
- Policy target = the move actually played (one-hot). Value target = the game
  result, from the side to move's perspective.
- **Memory fix:** the dataset library's streaming `.shuffle()` grew to ~10 GB
  of RAM and got the process OOM-killed on an 11 GB machine. It is replaced by
  shuffling the order of the ~26k monthly parquet shards and reading them
  sequentially, which stays under 1 GB. Each session uses a new seed, so
  resuming sees new games.
- **Parallel ingestion:** PGN parsing and board encoding were the bottleneck,
  so `--workers` processes (default 2) each stream a disjoint slice of the
  shuffled shards, while the main process only fills the buffer and trains.
  Throughput went from ~4,300 to ~6,900 positions/s; now the training step is
  the limit. The workers are forked before CUDA starts.
- **Leak containment:** the streaming stack (`datasets`/pyarrow) leaks about
  1 GB per hour per process, and after two hours it nearly ran the machine out
  of memory. Each worker is now a small supervisor that runs the stream in a
  fresh child process and replaces it every 30k games (~13 minutes). The child
  flushes its queue before exiting, so no games are lost. Replacing children
  much more often trips HuggingFace's API rate limit (1,000 requests per 5
  minutes), because each new child lists the dataset.
- **Faster encoding:** `encode_board` builds the 12 piece planes from
  python-chess bitboards in one numpy call instead of looping over pieces.
  It's 1.9x faster (27 → 14 µs), and its output is byte-identical to the old
  encoder on 38k positions.

**2. Self-play fine-tuning (`train.py` + `selfplay.py`).** This is the AlphaZero
loop:

- 64 games are played in parallel with 128 simulations per move. Moves are
  sampled in proportion to visit counts for the first 20 plies, then played
  greedily.
- Each position stores its MCTS visit distribution π as the policy target and
  the final game result z as the value target.
- Every 32 finished games, the net runs gradient steps proportional to the new
  data (each position is sampled about twice).

**Replay buffer (`replay_buffer.py`):** a 500k-position ring buffer (~600 MB
of RAM). Policy targets are stored *sparse* (legal-move indices plus fp16
probabilities) and densified per batch. Each block is flushed to compressed
`.npz` shards on disk.

**Built to be interrupted:** training runs on idle hours, so work happens in
blocks. Each block ends with an atomic checkpoint (tmp file + `os.replace`)
and a buffer flush. The first Ctrl+C stops cleanly within seconds. Rerunning
the script resumes the model, optimizer, GradScaler and buffer exactly.
Pretraining and self-play share the same checkpoint and buffer, so switching
phases just means running the other script.

### Playing (`engine.py`)

The play path is: opening book → Lichess 7-piece endgame tablebase (probed
online, only when ≤ 7 pieces remain) → MCTS with 6,400 simulations per move
(~5 s, batched 16 leaves per GPU call, tree reused between moves). The search
budget is doubled or quadrupled in endgames, where the tree is narrow and deep
calculation decides the game.

- **Opening book:** optional; run `alphazero/download_book.sh` to fetch the
  Komodo polyglot book (freely distributed, not stored in this repo). The
  engine picks among the strong book moves (weight ≥ half the best),
  weighted, so repeated games don't replay one opening.
- **UI:** after each engine move the status bar shows the search depth,
  seldepth, simulations run and visits reused. A pawn reaching the last rank
  opens a promotion picker, and every clicked move is checked for legality
  before it's played.

---

## Performance: making MCTS fast

Self-play takes 95%+ of the wall-clock time, so MCTS throughput decides how
much the engine can learn. Two optimizations came out of profiling while GPU
utilization sat at ~33%:

1. **Batched leaf evaluation.** With one network call per leaf, almost all of
   the time goes to per-call overhead (Python → CUDA launch → host/device
   copy) rather than computation. A forward pass takes **5.2 ms at batch 1
   and 5.4 ms at batch 64**. Collecting one leaf from each of 64 games per
   GPU call amortizes that overhead.
2. **Push/pop traversal.** After batching, profiling showed the time was
   dominated by python-chess legal-move generation, which the tree ran up to
   3 times per simulation. Switching to a push/pop search board generates it
   once per leaf and removes the per-simulation board copy.

```
$ python benchmarks/bench_mcts.py        # 64 games × 128 sims, pretrained net
variant                  sims/s  vs naive
unbatched + copy            159      1.0x
batched   + copy           2263     14.3x
batched   + push/pop       2538     16.0x

push/pop over batched+copy: +12%
```

Across runs: batching gives 14–17x, push/pop adds 10–15%, and the total is
16–18x (on an idle machine; measured while training was also running, the
ratios came out a little higher). A fourth row adds CUDA graphs, which gain
only a few percent at self-play batch sizes. `benchmarks/check_equivalence.py` verifies that the push/pop tree
produces **exactly the same visit counts and values** as the copy-based
baseline (`mcts_copy_baseline.py`), including checkmate, stalemate and
50-move-rule positions. It is a pure speedup with no change in behavior.

**Play-time search.** A game against one opponent has only one tree, so the
batching above doesn't apply. Three changes speed up that case:

- **CUDA graphs:** `Evaluator` captures one graph per power-of-two batch size
  and replays it. At small batches the GPU work is tiny and kernel-launch
  overhead dominates, so a batch-1 call drops from **~5.7 ms to ~1.0 ms**.
  The graph reads the weights in place, so it stays correct while training
  updates them.
- **Virtual-loss batching:** 16 leaves per GPU call instead of 1.
- **Tree reuse** between moves.

Measured over 20 middlegame plies (depth = length of the principal variation;
seldepth = deepest line searched):

| Setting | Time/move | Depth | Seldepth |
|---|---|---|---|
| 1 leaf per call, 400 sims, fresh tree (before) | 1.07 s | 12.2 | 13.8 |
| 16 leaves per call, 400 sims | 0.27 s | 9.9 | 11.5 |
| 16 per call, 1,600 sims + tree reuse | 1.30 s | 16.8 | 20.6 |
| 16 per call, 3,200 sims + tree reuse | 2.54 s | 17.6 | 21.4 |

In a short check (10 paired-opening games, a small sample), the new settings
at 1,600 simulations scored +7 =3 −0 against the old ones at 400.

---

## Results

Hardware: RTX 3070 Ti Laptop (8 GB), WSL2, Python 3.12, PyTorch 2.x.
Raw logs are in [`results/`](results).

**Supervised pretraining:** 1,780,611 Lichess games (both players ≥ 2000),
~137M positions, 267,752 steps of batch 512, in **~7.7 hours** of GPU time at
~5,500 positions/sec. Policy loss fell from 6.19 to 1.44 and value loss
settled around 0.70 (`results/pretrain_log.csv`).

**Pretrained net vs classical search** (200 MCTS simulations/move vs the
material alpha-beta searcher; small samples):

| Opponent | Result |
|---|---|
| Alpha-beta, depth 4 | **+5 =0 −0** (`results/match_vs_ab.log`) |
| Alpha-beta, depth 5 | +1 =0 −0 |

**Self-play fine-tuning:** ~8 hours of self-play starting from the pretrained
net produced ~5,800 games at ~11 games/min. In a 40-game arena, the fine-tuned
net scored **+5 =19 −16 (−98 Elo) against the pretrained net it started from**
(`results/arena.log`), so the pretrained weights stayed the active model.
The likely cause: 5.8k games of 128-simulation self-play is far too little
signal compared with 1.78M strong human games. The first updates mostly pulled
the net toward its own weaker play. At this compute scale, self-play is the
bottleneck. That is why the MCTS speed work matters, and why pretraining comes
first.

The shipped weights (`alphazero/weights/pretrained_128x10.pt`, fp16, 7 MB)
are that pretrained network.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # install a CUDA build of torch for GPU use

# Stage 1: minimax vs alpha-beta
python 1_search/search.py

# Stage 3: play against the pretrained net (tkinter UI)
cd alphazero && ./download_book.sh    # optional opening book
python ui.py
CHESS_MODEL=data/other.pt python ui.py   # play a specific checkpoint

# Benchmarks (from the repo root)
python benchmarks/check_equivalence.py
python benchmarks/bench_mcts.py
```

Training (run inside `alphazero/`; state lives in `alphazero/data/`):

```bash
python pretrain.py --minutes 480            # supervised on Lichess games
python pretrain.py --lr 3e-4                # override the learning rate saved in the checkpoint
python train.py --minutes 480               # self-play fine-tuning
python evaluate.py data/checkpoint.pt data/old.pt --games 40   # arena A vs B
python match_vs_ab.py 10 4                  # net vs alpha-beta depth 4, 10 games
```

To fine-tune from the shipped weights instead of from scratch:
`mkdir -p data && cp weights/pretrained_128x10.pt data/checkpoint.pt`.
Set `HF_TOKEN` for higher HuggingFace rate limits on long pretraining runs.

## Limitations and next steps

- Board operations run in Python (python-chess). A C++ move generator, or
  running self-play workers in several processes feeding one GPU batch, is
  the next big throughput lever.
- `evaluate.py` still evaluates one leaf at a time. Batching arena games the
  way `selfplay.py` does would speed up evaluation a lot.
- The learning rate is lowered by hand when the loss plateaus (1e-3 → 3e-4 →
  1e-4 so far). An automatic reduce-on-plateau schedule is the obvious next
  step.
- The network has no history planes, so it can't see repetitions itself;
  only the search can.
- The checkpoint records `{channels, blocks}`, so a larger network can be
  trained without breaking old checkpoints.
