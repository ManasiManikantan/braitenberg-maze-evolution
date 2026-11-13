"""AFTER: network-only evolution with the repo's new defaults.

Training mirrors evolution/experiment.py run_experiment (network_only=True,
evaluate_course_set_outcome, DEFAULT_ROBOT_NOISE, fresh noise seed per
evaluation, mutation 0.15/0.2, heading noise 0.05).
Usage: python after.py REPO OUT_DIR SEEDS GENERATIONS POP STEPS
"""

import sys
import time
from multiprocessing import Pool

REPO, OUT, SEEDS, GENS, POP, STEPS = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.experiment import (  # noqa: E402
    TRAINING_NOISE_SEED_BASE, MultiCourseWorld, evaluate_course_set_outcome)
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
import measure  # noqa: E402

COURSES = (1, 2, 3, 4)
EVAL_TRIALS = 5
NOISE = 0.05


def evaluate_genome(ga, genome, seed):
    controller = ga.decode_genome(genome)
    rows = []
    for d in COURSES:
        maze = Maze(difficulty=d)
        for trial in range(EVAL_TRIALS):
            w = World(maze=maze, vehicle=Vehicle(
                controller=controller, noise_stdev=NOISE,
                noise_seed=seed * 1000 + d * 10 + trial + 500000,
                navigation_heuristics=False, **DEFAULT_ROBOT_NOISE))
            w.run(STEPS)
            rows.append({"seed": seed, "course": d, "trial": trial, **measure.measure_world(w)})
    return rows


def run_seed(seed):
    ga = GeneticAlgorithm(population_size=POP, hidden_size=4, mutation_rate=0.15,
                          mutation_scale=0.2, seed=seed,
                          fitness_function=evaluate_course_set_outcome, network_only=True)
    noise_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, seed])
    log = open(f"{OUT}/after_progress_seed{seed}.txt", "w")
    start = time.time()

    def factory(controller):
        return MultiCourseWorld(controller, noise_stdev=NOISE,
                                noise_seed_base=int(noise_rng.integers(2**31)),
                                network_only=True)

    for g in range(GENS):
        best = ga.evaluate_population(world_factory=factory, steps=STEPS)
        h = ga.history[-1]
        comps = best.metrics["score_components"]
        goals = sum(c["reached_goal"] for c in best.metrics["courses"])
        h["gen_best_goals"] = goals
        h["gen_best_progress"] = comps["path_progress"]
        print(f"gen {g} best {h['generation_best_fitness']:.3f} mean {h['mean_fitness']:.3f} "
              f"goals {goals}/4 t={time.time() - start:.0f}s", file=log, flush=True)
        if g < GENS - 1:
            ga.next_generation()
    best = ga.best_individual
    rows = evaluate_genome(ga, best.genome, seed)
    hist = [{"seed": seed, **h} for h in ga.history]
    return rows, hist, best.genome


if __name__ == "__main__":
    with Pool(min(SEEDS, 10)) as p:
        results = p.map(run_seed, range(SEEDS))
    rows = [r for res in results for r in res[0]]
    hist = [h for res in results for h in res[1]]
    np.savez(f"{OUT}/after_genomes.npz", **{f"seed{s}": res[2] for s, res in enumerate(results)})
    summary = measure.write(rows, hist, OUT, "after")
    for k, v in summary.items():
        print("after", k, v)

    # Baseline genomes re-measured under the new (noisy) evaluation conditions.
    before = np.load(f"{OUT}/before_genomes.npz")
    ga = GeneticAlgorithm(population_size=4, hidden_size=4, seed=0)
    rows_b = [r for key in before.files
              for r in evaluate_genome(ga, before[key], int(key[4:]))]
    summary_b = measure.write(rows_b, hist, OUT, "before_under_new_noise")
    for k, v in summary_b.items():
        print("before@new-noise", k, v)
