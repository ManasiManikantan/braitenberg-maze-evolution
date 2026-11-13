"""Behavior/performance measures (evolution/metrics.py) for evolved seeds and wall followers.

Each controller runs on the same 100 random 12x12 mazes and starts (seed 4242,
separate from training/validation/test sets) with default robot noise.
Usage: python measure_table.py REPO OUT_CSV LABEL=GENOMES_NPZ [LABEL=GENOMES_NPZ ...]
"""

import csv
import sys
from multiprocessing import Pool

REPO, OUT_CSV = sys.argv[1:3]
SOURCES = [arg.split("=", 1) for arg in sys.argv[3:]]
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from controller.wall_follower import WallFollower  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from evolution.metrics import BEHAVIOR_MEASURES, PERFORMANCE_MEASURES, measure_controller, random_maze_episodes  # noqa: E402

EPISODES = random_maze_episodes(100, seed=4242, noise_seed_base=900000)


def job(spec):
    label, kind, payload = spec
    if kind == "wall":
        controller = WallFollower(payload)
    else:
        controller = GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked").decode_genome(payload)
    summary = measure_controller(controller, EPISODES)
    row = {"controller": label}
    for name in BEHAVIOR_MEASURES + PERFORMANCE_MEASURES:
        row[f"{name}_mean"] = round(summary["mean"][name], 3)
        row[f"{name}_std"] = round(summary["std"][name], 3)
    return row


if __name__ == "__main__":
    specs = []
    for label, path in SOURCES:
        genomes = np.load(path)
        specs += [(f"{label} seed{key[4:]}", "net", genomes[key]) for key in sorted(genomes.files)]
    specs += [("left wall follower", "wall", "left"), ("right wall follower", "wall", "right")]
    with Pool(10) as pool:
        rows = pool.map(job, specs)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'controller':26} {'speed':>12} {'clearance':>12} {'turn bias':>13} {'exploration':>12} "
          f"{'goal':>6} {'progress':>9} {'coll%':>6}")
    for r in rows:
        print(f"{r['controller']:26} "
              + " ".join(f"{r[f'{n}_mean']:+6.2f}±{r[f'{n}_std']:.2f}".rjust(12 if n != 'turning_bias' else 13)
                         for n in BEHAVIOR_MEASURES)
              + f" {r['goal_reached_mean']:6.2f} {r['path_progress_mean']:9.2f} {r['pct_steps_in_collision_mean']:6.0f}")
