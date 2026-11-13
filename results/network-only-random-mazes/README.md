# Network-only evolution on random mazes (60-degree side sensors)

Trained on 3 freshly generated perfect mazes per evaluation (20 x 20, 4-unit
corridors, random start cell and heading, per-maze limit 2 x shortest route at
0.05 units/step), 5 seeds x 60 generations x population 30. Side sensors at
+/-60 degrees. Final controller per seed chosen on a fixed 20-maze validation
set. Outcome-only fitness, physics and noise unchanged; no goal input.

`summary.csv` compares these controllers (`random_mazes`) with the
time-pressure controllers (`time_pressure`, trained on courses 1-4 with
parallel sensors) on 50 random test mazes never used in training or
validation, and on courses 1-4 (held out; 5 trials each).

Reference: untrained random networks (60-degree sensors, +0.5 output bias)
reach the goal on 0.9% of the 50 test mazes.

- `trajectories_random_mazes_seed3.png`, `..._seed0.png`: one controller on two
  random test mazes and on courses 1 and 2.
- `start_frame_overlay.png`: first 300 steps on all 50 test mazes in the
  robot's start frame, time-pressure vs. random-maze controller.
