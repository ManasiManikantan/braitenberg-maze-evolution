# MAP-Elites over turning bias x speed (locked simple-12x12 random-mismatch setup)

Run with `python -m evolution.network_map_elites OUT --seed-genomes
results/twenty-seeds/evolved_genomes.npz --evaluations 10000 --workers 9`.
`evolution/map_elites.py` was not changed.

## Setup

- **Controllers:** network-only, 5 inputs, hidden size 4.
- **Fitness:** outcome-only.
- **Robot:** default robot noise with a random-side wheel mismatch.
- **Mazes:** random 12x12 mazes with the 8x time limit.
- **Mutation:** rate 0.15, scale 0.2, crossover 0.8.
- **Grid:** 10 x 10 over turning bias (-1..1) x speed (0.35..0.63).
  - The speed range is the 5th-95th percentile of the 20 GA winners' speeds,
    widened by 25% on each side.
  - Values outside the range fall into the edge rows.
  - Speed = mean |forward speed| / max, so reversing counts as moving.
- **Batches:** each batch has 45 children, all evaluated on the same 8 random
  mazes and noise seed. 9 workers.
- **Start:** the 20 GA winners. There was no per-generation map to start from,
  because the GA run saved only its final winners.
- **Parent selection:** an elite's weight is
  (1 + empty neighbouring cells) / (1 + times its cell was improved).
- **Checkpoints:** written every 450 evaluations (`checkpoint.pkl`,
  `archive.npz`). Continue a run with `--resume`.
- **Run time:** 10,000 evaluations in 53 minutes, about 190-240 evaluations per
  minute. The GA ran at about 140 per minute on 10 workers.
- **Final check:** every elite was re-measured on the 100-maze measurement set
  (seed 4242) and on courses A, B and C (10 trials each). The script is
  `scripts/remeasure_elites.py`.
- **Elite stats:** each elite also logs reverse fraction, % steps in collision,
  exploration, wall side and light reliance. Light reliance is the mean change
  in the two motor outputs when the light inputs are set to 0 on the states the
  robot visited.

Files:
- `elites_remeasured.csv`: one row per elite.
- `who_wins_where.csv`.
- `report.txt`.
- `progress.txt`.
- `archive_success.png`.
- `trajectories.png`.

The full log of 10,000 evaluations (8 MB) was not committed.

## Results

**Filling the grid.** All 100 cells were filled by evaluation 1,959. In the
final archive, 99 elites are mutants and 1 is an original GA winner. Cells were
improved 370 times in total.

**Success is spread across the grid.** On the 100 mazes, 97 elites reach the
goal at least 50% of the time and 47 at least 70% of the time. The best elite
reaches 83%. Slow, medium and fast styles all solve mazes:

| speed | elites | best | elites >= 0.70 |
| --- | --- | --- | --- |
| slow | 35 | 0.77 | 11 |
| medium | 45 | 0.83 | 26 |
| fast | 20 | 0.80 | 10 |

Speed correlates only weakly with success (Spearman rho = +0.20).

The weakest cells are at the slow edge with strong turning, at 0.41-0.58.

**Who wins where.** Elites are grouped by their re-measured bias and speed.
The table shows mean success rates.

| bias | speed | n | random | A | B | C |
| --- | --- | --- | --- | --- | --- | --- |
| right | slow | 12 | 0.61 | 0.02 | 0.83 | 0.96 |
| right | medium | 16 | 0.69 | 0.04 | 0.88 | 0.86 |
| right | fast | 10 | 0.64 | 0.14 | 0.81 | 0.97 |
| neutral | slow | 12 | 0.67 | 0.62 | 0.53 | 0.92 |
| neutral | medium | 12 | 0.76 | 0.62 | 0.52 | 0.96 |
| neutral | fast | 0 | - | - | - | - |
| left | slow | 11 | 0.61 | 0.62 | 0.07 | 0.95 |
| left | medium | 17 | 0.70 | 0.69 | 0.18 | 0.95 |
| left | fast | 10 | 0.70 | 0.93 | 0.00 | 0.96 |

- **Neutral elites are the new finding.** Medium-speed neutral elites have the
  best random-maze success (0.76) and solve *both* A and B about 60% of the time.
- The top elites, at 0.83, are neutral. Elite 53 solves A 8/10, B 8/10 and C 10/10.
- The GA never produced this style. Its 4 neutral seeds were weaker.
- Turning direction still decides A vs B. Right-turners solve B, and left-turners solve A.
- Fast left-turners solve A 93% of the time.

**Light reliance is low everywhere.** It averages 0.067 and ranges 0.01-0.18.
It does not predict success (rho = +0.03).

**Caveat: descriptors measured on 8 mazes are noisy.** Re-measured on 100
mazes, 34 of the 100 elites land in a different bias column than the one they
were filed under. The heatmap places elites by their search cell. The table
above groups them by their 100-maze measurements.

`trajectories.png` shows three different elites on the same maze:

| elite | style | bias | speed | steps to goal |
| --- | --- | --- | --- | --- |
| 78 | fast left-turner | +0.65 | 0.57 | 319 |
| 46 | neutral | -0.09 | 0.48 | 777 |
| 5 | slow right-turner | -0.82 | 0.41 | 2170 |
