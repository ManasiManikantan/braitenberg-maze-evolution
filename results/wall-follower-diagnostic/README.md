# Hand-written wall followers (diagnostic, no evolution)

`scripts/wall_follower.py`: left- and right-wall followers using only the
three distance sensors (-60/0/+60 degrees; no light), network-only physics,
default robot noise + heading noise 0.05. Rules: pivot away when the front
ray is short, veer away when the followed side is too close, arc toward it
(radius ~2 units) when it reads far, otherwise drive straight. Tuned only on
the 20 validation mazes (seed 777).

Test sets: 50 random mazes at each curriculum size (12x12, 16x16, 20x20;
seed 2026, the same 20x20 set as the evolution runs) and courses 1-4 x 5
trials. Each run goes up to 20,000 steps; "within current limit" is read
from the same run (limit = 2 x shortest route / 0.05).

Failure labels (no-limit runs): `stuck` = moved < 1 unit in the last 2,000
steps; `loop` = no new 1-unit cell in the last 5,000 steps; `still exploring
at cap` otherwise. Under the current limit, runs that reach the goal later
are `time-out (goal reached later)`.

Layout comparison (from `scripts/wf_gaps.py` and the shortest-route profile):

| Layout | Footprint | Shortest route | Separate solids | Narrowest gaps between solids |
| --- | --- | --- | --- | --- |
| random 12x12 | 3x3 cells | 8 units median (2.0 cells) | 1 (all walls connected) | none; corridors 4.0 |
| random 16x16 | 4x4 cells | 12 units median (3.0 cells) | 1 | none; corridors 4.0 |
| random 20x20 | 5x5 cells | 17.5 units median (4.4 cells) | 1 | none; corridors 4.0 |
| course 1 | 20x20 (5x5 cells) | 15.0 units (3.8 cells) | 3 | 2.0, 2.5 |
| course 2 | 20x20 | 15.0 units (3.8 cells) | 8 | 0.5, 0.5, 1.0, 1.5 (impassable), 5 gaps of 2.0-3.0 |
| course 3 | 20x20 | 18.5 units (4.6 cells) | 4 | 0.7 (impassable), 3.0 |
| course 4 | 20x20 | 15.0 units (3.8 cells) | 6 | 1.0, 1.0 (impassable), 2.0, 3.0 |
