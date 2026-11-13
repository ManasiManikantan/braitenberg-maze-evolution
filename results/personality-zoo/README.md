# Personality zoo

Warm-started evolution in more worlds, in the locked simple-12x12
random-mismatch setup:
- **Warm start:** parent + 29 mutated copies.
- **Length:** 20 generations.
- **Runs:** 10, all in parallel (19 minutes).

Every final robot and its parent are measured in the same plain world on 50
mazes (`scripts/measure_zoo.py`). That table is combined with the danger runs
in `../danger-worlds`.

## Worlds and runs

| World | Parents | Settings |
| --- | --- | --- |
| safe (control) | seeds 3, 2 | the plain locked world |
| battery | seeds 3, 2 | `World(energy = 100 x shortest-route length)`; each step costs \|L\| + \|R\| |
| explorer | seeds 3, 2 | hidden light (6 units) + goal moved to a random dead end (`with_goal_in_dead_end`) |
| fear: safe | fear robot x 2 | plain world |
| fear: walls hurt | fear robot x 2 | health 15, 1 per new hit (the version with the loophole) |

**Battery calibration** (`scripts/calibrate_zoo.py`).
- A perfect straight drive costs 30 energy per unit of route.
- The GA winners' successful runs use a median of 58-66 per unit.
- The budget is 100 per unit.

**Fear robot** (`fear_seed.npz`, `scripts/fear_seed.py`).
- A hand-set network that centres itself in corridors, slows and turns away
  when something is close in front, and seeks the light weakly.
- On 50 calibration mazes it reaches the goal 40% of the time.
- Its clearance is 0.23, against 0.14-0.15 for the GA winners.
- Its speed is 0.20, and it touches a wall in 40% of steps.

The generation-best genome of every run is in `best_per_generation.npz`.

## Results

Full tables are in `report.txt` and `fingerprints.csv`.

**Safe control.** It barely changed:
- Seed 3's validated winner was the unmutated parent.
- Seed 2 changed by -0.01 in speed and -3.4 points in % touching.

So the differences between worlds come from the worlds, not from the extra
generations.

**Explorer.** Changes relative to the safe control:
- Wall contact rose by 17.7 points.
- Exploration rose by 0.03.
- Seed 3's turning bias fell from +0.85 to +0.42.

**Walls hurt (with the loophole) and hurt + hidden light.** Both made robots
faster and made them touch walls more (+8 and +12 points).

**Battery.** Almost no change; energy per step did not fall.

**Turning direction.** No lineage flipped its turning direction.

**Fear faded in both worlds** (`fear_over_generations.png`).
- In both safe runs, the clearance of each generation's best fell from
  about 0.20 to 0.11.
- In walls-hurt, one run held 0.17 and the other fell to 0.13.
- Goal success rose from 0.52 to 0.72-0.90 as the robots sped up.

`fingerprint.png` shows each robot's change from its parent on 8 measures,
grouped by world. `trajectories.png` shows one robot per world on the same
maze.
