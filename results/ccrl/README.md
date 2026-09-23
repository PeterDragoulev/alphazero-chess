# CCRL-anchored matches

fastchess, 40 moves in 2 minutes (repeating), 8-move opening suite in random
order, 2 games at a time on one RTX 3070 Ti laptop (the GPU is shared by both
of our engine instances). Our engine: `alphazero/uci_engine.sh`, no own book,
no tablebases. Anchors: Stash, whose versions have CCRL 40/15 ratings.

| Log | Our net | Opponent (CCRL 40/15) | Result | Elo diff | Implied |
|---|---|---|---|---|---|
| match_stash23.0_40_120 | 6.5M-game Lichess 2400+ net | Stash 23.0 (2902) | +3 =0 −2 | +70 | ~2970 |
| match_stash27.0_40_120 | same | Stash 27.0 (3022) | +1 =2 −2 | −70 | ~2950 |

Five games per match gives error bars of roughly ±300 Elo each; together they
point to **~2,950 CCRL 40/15**, consistent with the Stockfish `UCI_Elo`
estimate (~2,850–2,900, `results/stockfish/`).
