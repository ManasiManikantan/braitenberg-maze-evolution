"""BEFORE: network-only evolution using the repo's code as it was at e43fee1.

Same pipeline as evolution/experiment.py (evaluate_course_set fitness,
mutation 0.15/0.2, heading noise 0.05) but with navigation_heuristics=False,
since the repo has no network-only training option yet.
Usage: python baseline.py REPO OUT_DIR SEEDS GENERATIONS POP STEPS
"""

import sys
from multiprocessing import Pool

REPO, OUT, SEEDS, GENS, POP, STEPS = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.experiment import evaluate_course_set  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.robot import Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
import measure  # noqa: E402

COURSES = (1, 2, 3, 4)
EVAL_TRIALS = 5
NOISE = 0.05


class NetOnlyMultiCourse:
    def __init__(self, controller):
        self.worlds = [
            (f"course_{d}", World(maze=Maze(difficulty=d), vehicle=Vehicle(
                controller=controller, noise_stdev=NOISE, noise_seed=900000 + d,
                navigation_heuristics=False)))
            for d in COURSES
        ]

    def run(self, steps):
        self.steps = steps
        for _, w in self.worlds:
            w.run(steps)


def run_seed(seed):
    ga = GeneticAlgorithm(population_size=POP, hidden_size=4, mutation_rate=0.15,
                          mutation_scale=0.2, seed=seed, fitness_function=evaluate_course_set)
    best = ga.run(generations=GENS, steps=STEPS, world_factory=NetOnlyMultiCourse)
    controller = ga.decode_genome(best.genome)
    rows = []
    for d in COURSES:
        maze = Maze(difficulty=d)
        for trial in range(EVAL_TRIALS):
            w = World(maze=maze, vehicle=Vehicle(controller=controller, noise_stdev=NOISE,
                                                 noise_seed=seed * 1000 + d * 10 + trial,
                                                 navigation_heuristics=False))
            w.run(STEPS)
            rows.append({"seed": seed, "course": d, "trial": trial, **measure.measure_world(w)})
    hist = [{"seed": seed, **h} for h in ga.history]
    return rows, hist, best.genome


if __name__ == "__main__":
    with Pool(min(SEEDS, 10)) as p:
        results = p.map(run_seed, range(SEEDS))
    rows = [r for res in results for r in res[0]]
    hist = [h for res in results for h in res[1]]
    np.savez(f"{OUT}/before_genomes.npz", **{f"seed{s}": res[2] for s, res in enumerate(results)})
    summary = measure.write(rows, hist, OUT, "before")
    for k, v in summary.items():
        print(k, v)
