# Network-only evolution with per-course time limits

| Condition in `summary.csv` | What it is |
| --- | --- |
| `no_time_pressure` | Controllers from `../network-only-before-after` (trained and evaluated with 3,000 steps on every course), re-measured with identical trials to add pinned-stretch statistics. |
| `no_time_pressure_controllers_at_limits` | The same controllers evaluated under the new per-course limits (not retrained). |
| `time_pressure` | Retrained with per-course limits: 2 x shortest path / 0.05 units per step = 600, 600, 740, 600 steps on courses 1-4. 5 seeds x 60 generations x population 30. |

Everything else is identical: network only, outcome-only fitness, +0.5 output
bias, default robot noise, unchanged collision physics. Best controller per
seed, 5 noisy trials per course.

`longest_pinned` is the longest run of consecutive steps in collision.
`course2_where_stuck.png` overlays all 25 course-2 evaluation runs.
