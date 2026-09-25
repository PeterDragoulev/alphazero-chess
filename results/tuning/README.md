# Search tuning logs

Output of `alphazero/tune_match.py`: two engine configurations (or nets) play
each other at fixed simulations per move, 128 games (64 openings from the
8-move suite, each played with both colours), Elo with a 95% CI computed over
opening pairs. "B" is the candidate, "A" the baseline.

| Log | What |
|---|---|
| round1 | policy_temp 1.3, play_batch 32, c_puct 2.0 vs defaults (800 nodes) |
| round2 | around policy_temp 1.3: 1.6, c_puct 1.2, FPU 0.4 |
| round3 | policy_temp 1.3 vs 1.0 at 3,200 nodes |
| batch8 | play_batch 8 vs 16 at equal nodes |
| se_check / se_final | SE net vs pre-SE net, mid-training and after a night |
| speed | sims/s by batch size, normal vs frozen evaluator (idle GPU) |
| batch_equal_time | batch 8 and 32 vs 16 at equal time (nodes scaled by speed) |
| batch64 | batch 64 vs 32 at equal time |
| contempt | draw contempt 0.1 |
| round4 | root FPU −1 / 0, Q-based final move choice |
| round5 | eval cache, solver, smart time use (node clock), c_puct growth |
| round6_qsel | Q-based final move choice at 3,200 nodes |
