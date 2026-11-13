# Twenty seeds of the locked simple-12x12 setup (random-side mismatch)

Same setup as `../simple-12x12-random-mismatch`, with 20 seeds (0-19) x 100
generations x population 30 instead of 5 x 60. Seeds 0-4 were rerun at 100
generations so all 20 are comparable; they are therefore different
controllers from the 60-generation seeds 0-4.

- Behavior measures (`evolution/metrics.py`): 100 random 12x12 mazes, maze seed
  4242, noise seeds 900000+.
- Goal success: 50 random test mazes (seed 2026) plus courses A/B/C x 10 trials,
  8x time limit.
- Turner classes: right if bias < -0.3, left if > +0.3, neutral otherwise.

Files:
- `per_seed.csv`: one row per seed.
- `measures.csv`: raw metric means and stds, plus both wall followers.
- `test_episodes.csv`, `history.csv`.
- `evolved_genomes.npz`, `untrained_genomes.npz`.
- `analysis.txt`: full printout, including the permutation tests.
- `bias_vs_AB.png`.
- `scripts/`: `after_simple.py REPO OUT 20 100 30`, `measure_table.py`, `analyze_20.py`.

## Results

| class | seeds | random | A | B | C | A-B |
| --- | --- | --- | --- | --- | --- | --- |
| right (12) | 1 2 6 7 9 10 11 12 13 14 15 16 | 0.70 | 0.07 | 0.96 | 0.88 | -0.89 |
| neutral (4) | 0 8 17 18 | 0.67 | 0.28 | 0.62 | 0.90 | -0.35 |
| left (4) | 3 4 5 19 | 0.68 | 0.85 | 0.17 | 0.97 | +0.68 |

- **Turning bias predicts course preference.** Turning bias vs (A - B) has
  Spearman rho = +0.83 (permutation p < 0.001). Right-turners solve B, and
  left-turners solve A.
- **Left-turners are not reliably worse.** Two-sided permutation tests,
  left vs right turners:

  | measure | left | right | p |
  | --- | --- | --- | --- |
  | random-maze success | 0.68 | 0.70 | 0.45 |
  | course C | 0.98 | 0.88 | 0.60 |
  | % steps in collision | 24.0 | 21.4 | 0.61 |
  | better of A/B | 0.93 | 0.97 | 0.68 |

  The earlier "left-turners fail" impression came from seeds 3 and 4 of the
  60-generation run, which looped on both courses. At 100 generations, seeds
  3, 5 and 19 solve A 10/10.
- **The handedness split is lopsided:** 12 right vs 4 left. Under a 50/50
  split, a 12-4 or more extreme result happens with two-sided binomial p = 0.08.
  That is weak evidence, and a 20-seed sample cannot settle it.
- **Wall side does not separate the classes.** Most seeds sit at 0.4-0.75,
  with no clean left/right split like the wall followers' 0.78 / 0.22. Seed 4
  (a weak left-turner, wall side 0.20) is the only clear right-wall hugger.
- **Seed 16 drives backwards** 79% of the time. Its signed speed is -0.31 of
  maximum, so its `speed` measure (which counts reversing as 0) reads 0.01. It
  still solves B and C 10/10. Its heading and its direction of travel turn the
  same way, so the "right-turner" label still holds. Its sensors, however,
  point away from where it is going.
