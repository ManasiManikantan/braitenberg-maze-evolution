# Simple setup: shared random 12x12 mazes, 8x time limit, held-out courses A/B/C

Network-only evolution, 5 seeds x 60 generations x population 30. Each
generation the whole population runs the same 8 random 12x12 loop-free mazes
(and noise seed); per-maze limit 8 x shortest route at 0.05 units/step;
outcome-only fitness with the stuck penalty; 3 distance sensors (0, +/-60
deg) + 2 light sensors (walls block); default robot noise; physical motor
convention (a faster right wheel turns left). Final controller per seed
chosen on 20 fixed validation mazes.

Tests (never trained on): 50 random 12x12 test mazes (seed 2026) and courses
A, B, C (`sim/held_out_courses.py`) x 10 noisy trials, compared with
untrained networks and the hand-written left/right wall followers
(`scripts/wall_follower.py`) under the same 8x limit
(`wall_followers_simple.csv`).

`turning_bias_and_light_wiring.csv`: median (right - left) motor command on
visited states (positive = turns left) and the light-to-motor sensitivities
with their Braitenberg labels (physical convention, so classic labels apply).
