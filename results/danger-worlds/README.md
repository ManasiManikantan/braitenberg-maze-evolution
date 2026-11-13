# Dangerous worlds: walls hurt, and walls hurt + hidden light

Warm-started evolution in the locked simple-12x12 random-mismatch setup.

## Worlds

Fitness stays outcome-only in both worlds.

- **A, walls hurt:** `World(health=15)`. Each new collision (entering
  contact) costs 1. At 0 the run ends without the goal; the robot keeps its
  path progress.
  - At health 15, the hand-written wall followers survive 86% (left) and
    72% (right) of runs (`scripts/calibrate_health.py`).
- **B, walls hurt + hidden light:** as A, and
  `Vehicle(light_visible_distance=6)`, so the light sensors read 0 beyond
  6 units. Walls still block the light.

## Runs

- **Parents:** GA winners 3 and 19 (left-turners), and 2 and 13 (right-turners).
- **Population:** 30 = the winner + 29 mutated copies.
- **Length:** 30 generations, 8 runs in parallel (34 minutes).
- **Final robot:** best on 20 validation mazes in its own world.

## Measurement

`scripts/measure_danger.py` runs each original and evolved robot on 50 mazes
in all three worlds: baseline, A and B. The outputs are `measures.csv`,
`before_after.csv` and `report.txt`.

## Result: not more careful

Changes are measured in each robot's new world.

**Hits and scraping.**
- Hits per run fell by 0.6 in both worlds.
- But the share of steps touching a wall *rose*, by 6 points in A and 9 in B.
- Speed rose for most robots, and clearance barely changed.
- The cause is a loophole: only new contacts cost health, so scraping along
  a wall was free.
- `World(contact_step_cost=...)` closes it. The re-run is in
  `../walls-hurt-v2`.

**Turning direction.** Biases moved toward neutral (seed 3 from +0.85 to
+0.62, seed 2 in B from -0.88 to -0.53). None flipped.

**Hidden light.** All 4 robots evolved in B explore more than their parents.
Measured in world B, they also explore more than the robots evolved in A.

Caution: these runs had 30 generations, while the zoo's safe control
(`../personality-zoo`) had 20.
