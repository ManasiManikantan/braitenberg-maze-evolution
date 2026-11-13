"""Re-measure the saved baseline genomes with the fixed measure.py.

Uses the original code snapshot and the exact same evaluation conditions and
noise seeds as baseline.py (heading noise 0.05, no robot noise, network only).
Usage: python remeasure_before.py BEFORE_REPO OUT_DIR
"""

import csv
import sys

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.robot import Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
import measure  # noqa: E402

STEPS = 3000
genomes = np.load(f"{OUT}/before_genomes.npz")
ga = GeneticAlgorithm(population_size=4, hidden_size=4, seed=0)
rows = []
for key in genomes.files:
    seed = int(key[4:])
    controller = ga.decode_genome(genomes[key])
    for d in (1, 2, 3, 4):
        maze = Maze(difficulty=d)
        for trial in range(5):
            w = World(maze=maze, vehicle=Vehicle(controller=controller, noise_stdev=0.05,
                                                 noise_seed=seed * 1000 + d * 10 + trial,
                                                 navigation_heuristics=False))
            w.run(STEPS)
            rows.append({"seed": seed, "course": d, "trial": trial, **measure.measure_world(w)})
history = list(csv.DictReader(open(f"{OUT}/before_history.csv")))
summary = measure.write(rows, history, OUT, "before")
for k, v in summary.items():
    print(k, {m: round(x, 3) for m, x in v.items()})
