# CCRL-anchored matches

fastchess, 40 moves in 2 minutes (repeating), 8-move opening suite in random
order, 2 games at a time on one RTX 3070 Ti laptop (the GPU is shared by both
of our engine instances). Our engine: `alphazero/uci_engine.sh`, no own book,
no tablebases. Anchors: Stash, whose versions have CCRL 40/15 ratings.

| Log | Our net | Opponent (CCRL 40/15) | Result | Elo diff | Implied |
|---|---|---|---|---|---|
| match_stash23.0_40_120 | 6.5M-game Lichess 2400+ net | Stash 23.0 (2902) | +3 =0 −2 | +70 | ~2970 |
| match_stash27.0_40_120 | same | Stash 27.0 (3022) | +1 =2 −2 | −70 | ~2950 |
| match_ft_stash27.0_40_120 | + master-games fine-tune | Stash 27.0 (3022) | +2 =2 −1 | +70 | ~3090 |
| match_ft_vs_pre_40_120 | fine-tuned vs pre-fine-tune net (head to head) | — | +3 =0 −2 | +70 | |
| match_native_stash27.0_40_120 | + Stockfish-eval value targets, native C++ tree (7 games) | Stash 27.0 (3022) | +2 =3 −2 | 0 | ~3020 |
| match_se_stash27.0_40_120 | + SE blocks, policy temp 1.3, batch 32, frozen evaluator (7 games) | Stash 27.0 (3022) | +4 =1 −2 | +102 | ~3120 |

The fine-tune is `alphazero/finetune_otb.sh`: 150 minutes at lr 3e-5 on
over-the-board games with both players 2500+ (classical), mixed 50/50 with
Lichess 2400+; both of its matches came out 3-2 in its favour.

Five to seven games per match gives error bars of roughly ±250-300 Elo each.
The earlier nets land around 2,950-3,000; the current net and search around
**3,000-3,100 CCRL 40/15**. (The Stockfish `UCI_Elo` estimate of ~2,850-2,900 in
`results/stockfish/` was measured with an earlier net.)
