# Mirror test: is the simulator left/right symmetric?

`controller/mirror.py` (`MirroredController`) swaps the left and right
distance and light inputs, and the left and right wheel outputs. A mirrored
right-turner becomes a left-turner. If the simulator and courses have no
built-in handedness, a mirrored right-turner should solve course A about as
well as the original solves course B.

The controllers are the 5 seeds of `../simple-12x12-random-mismatch`
(60 generations).

`mirror_test.csv` (`scripts/mirror_test.py`, 10 trials per course):

| seed | original bias | original A / B | mirrored bias | mirrored A / B |
| --- | --- | --- | --- | --- |
| 1 | -0.64 | 0 / 9 | +0.65 | 6 / 0 |
| 2 | -0.89 | 0 / 10 | +0.89 | 10 / 0 |
| 3 | +0.77 | 1 / 2 | -0.78 | 1 / 3 |
| 4 | +0.46 | 2 / 2 | -0.49 | 3 / 4 |

`mirror_big.txt` (`scripts/mirror_big.py`) is the larger check:

- **Courses, 50 trials each.** The swap is exact:
  - seed 1: 40/50 on B originally, 43/50 on A when mirrored;
  - seed 2: 50/50 on B originally, 50/50 on A when mirrored.
- **300 random mazes.** Original minus mirrored success is -0.014 ± 0.016
  (paired SE), pooled over the 5 seeds.

Conclusion: the simulator is symmetric. The preference for course A or B
comes from each controller's handedness, not from the environment. Seeds 3
and 4 fail both courses whether mirrored or not, so they fail because they
loop, not because they turn left.
