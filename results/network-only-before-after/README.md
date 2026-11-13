# Network-only evolution: baseline vs. new defaults (no time pressure)

| Condition in `summary.csv` | What it is |
| --- | --- |
| `before` | Original code (e43fee1) with the hand-coded rules off: old shaped fitness, zero output biases, heading noise 0.05 only. 5 seeds x 15 generations x population 30. |
| `before_under_new_noise` | The same `before` controllers re-measured under the new sensor/motor/start noise. |
| `no_time_pressure` | Network-only defaults (outcome-only fitness, +0.5 output bias, robot noise), 5 seeds x 60 generations x population 30, **3,000 steps on every course** (no per-course time limit). |

All conditions: best controller per seed, 5 noisy trials per course, 3,000-step
episodes. Path progress = drop in shortest-path (BFS) distance to the goal
divided by the starting distance.

Finding: without time pressure, evolved controllers reach the goal (39%) but
spend 53% of steps pinned against walls, pivoting in place, because wall
contact costs only time and time is not scored. See `trajectories_seed*.png`.
