# Simple-12x12 baseline rerun with a random-side motor mismatch

Identical to `../simple-12x12` (the locked baseline) except that the 5%
wheel mismatch now has a random sign per run (`randomize_mismatch_side`,
on in `DEFAULT_ROBOT_NOISE`). 5 seeds x 60 generations.

Why: with the fixed mismatch (left wheel always stronger) the hand-written
right-wall follower beat the left one (72% vs 54% of the random 12x12 test
mazes). `scripts/wf_mismatch.py` output, goal success within the 8x limit:

| Mismatch | random L / R | course A L / R | course B L / R |
| --- | --- | --- | --- |
| +0.05 (old, left stronger) | 54% / 72% | 80% / 60% | 20% / 90% |
| 0 | 70% / 64% | 90% / 50% | 20% / 80% |
| -0.05 (right stronger) | 76% / 60% | 90% / 0% | 40% / 90% |
| random side (new) | 74% / 68% | 100% / 30% | 30% / 100% |

Files: `summary.csv` (evolved vs untrained vs wall followers on 50 random
12x12 test mazes and courses A/B/C), `turning_bias_and_light_wiring.csv`,
`measures.csv` (evolution/metrics.py measures for the baseline seeds, these
seeds and both wall followers on 100 random 12x12 mazes, seed 4242),
`left_turners_on_AB.png` (seeds 3 and 4 on courses A and B, 3 trials each).
