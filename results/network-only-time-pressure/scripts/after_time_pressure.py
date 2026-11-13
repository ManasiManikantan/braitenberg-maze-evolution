"""TIME PRESSURE: network-only evolution with per-course step limits.

Training mirrors evolution/experiment.py run_experiment with its defaults
(network_only=True, evaluate_course_set_outcome, DEFAULT_ROBOT_NOISE, fresh
noise seed per evaluation, mutation 0.15/0.2, heading noise 0.05, per-course
limits = 2 x shortest path / 0.05). Same evaluation noise seeds as after.py.

Writes:
  time_pressure_*                     new controllers, per-course limits
  no_time_pressure_remeasured_*       previous controllers, 3,000 steps (adds pinned stats)
  no_time_pressure_at_limits_*        previous controllers under the new limits
Usage: python after_time_pressure.py REPO OUT_DIR PREV_GENOMES SEEDS GENERATIONS POP
"""

import sys
import time
from multiprocessing import Pool

REPO, OUT, PREV = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS, GENS, POP = map(int, sys.argv[4:7])
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.experiment import (  # noqa: E402
    TRAINING_NOISE_SEED_BASE, MultiCourseWorld, course_step_limit, evaluate_course_set_outcome)
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
import measure  # noqa: E402

COURSES = (1, 2, 3, 4)
EVAL_TRIALS = 5
NOISE = 0.05
STEP_CAP = 3000


def evaluate_genome(genome, seed, use_limits, keep_paths=False):
    ga = GeneticAlgorithm(population_size=4, hidden_size=4, seed=0)
    controller = ga.decode_genome(genome)
    rows, paths = [], []
    for d in COURSES:
        maze = Maze(difficulty=d)
        limit = course_step_limit(maze) if use_limits else None
        steps = STEP_CAP if limit is None else min(STEP_CAP, limit)
        for trial in range(EVAL_TRIALS):
            w = World(maze=maze, vehicle=Vehicle(
                controller=controller, noise_stdev=NOISE,
                noise_seed=seed * 1000 + d * 10 + trial + 500000,
                navigation_heuristics=False, **DEFAULT_ROBOT_NOISE))
            row = measure.run_and_measure(w, steps)
            rows.append({"seed": seed, "course": d, "trial": trial, "step_limit": steps, **row})
            if keep_paths:
                paths.append((seed, d, trial, np.array(w.position_history)))
    return rows, paths


def run_seed(seed):
    ga = GeneticAlgorithm(population_size=POP, hidden_size=4, mutation_rate=0.15,
                          mutation_scale=0.2, seed=seed,
                          fitness_function=evaluate_course_set_outcome, network_only=True)
    noise_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, seed])
    log = open(f"{OUT}/time_pressure_progress_seed{seed}.txt", "w")
    start = time.time()

    def factory(controller):
        return MultiCourseWorld(controller, noise_stdev=NOISE,
                                noise_seed_base=int(noise_rng.integers(2**31)),
                                network_only=True)

    for g in range(GENS):
        best = ga.evaluate_population(world_factory=factory, steps=STEP_CAP)
        h = ga.history[-1]
        goals = sum(c["reached_goal"] for c in best.metrics["courses"])
        h["gen_best_goals"] = goals
        h["gen_best_progress"] = best.metrics["score_components"]["path_progress"]
        print(f"gen {g} best {h['generation_best_fitness']:.3f} mean {h['mean_fitness']:.3f} "
              f"goals {goals}/4 t={time.time() - start:.0f}s", file=log, flush=True)
        if g < GENS - 1:
            ga.next_generation()
    best = ga.best_individual
    rows, paths = evaluate_genome(best.genome, seed, use_limits=True, keep_paths=True)
    hist = [{"seed": seed, **h} for h in ga.history]
    return rows, hist, best.genome, paths


def evaluate_previous(args):
    key, genome, use_limits = args
    return evaluate_genome(genome, int(key[4:]), use_limits)[0]


if __name__ == "__main__":
    prev = np.load(PREV)
    prev_jobs = [(k, prev[k], False) for k in prev.files] + [(k, prev[k], True) for k in prev.files]
    with Pool(10) as p:
        new_async = p.map_async(run_seed, range(SEEDS))
        prev_results = p.map(evaluate_previous, prev_jobs)
        results = new_async.get()

    n = len(prev.files)
    prev_hist = [{"seed": "-", "note": "see no_time_pressure history"}]
    for name, chunk in (("no_time_pressure_remeasured", prev_results[:n]),
                        ("no_time_pressure_at_limits", prev_results[n:])):
        summary = measure.write([r for rows in chunk for r in rows], prev_hist, OUT, name)
        print(name, {k: {m: round(x, 3) for m, x in v.items()} for k, v in summary.items()})

    rows = [r for res in results for r in res[0]]
    hist = [h for res in results for h in res[1]]
    np.savez(f"{OUT}/time_pressure_genomes.npz", **{f"seed{s}": res[2] for s, res in enumerate(results)})
    np.savez_compressed(f"{OUT}/time_pressure_eval_paths.npz",
                        **{f"seed{s}_c{d}_t{t}": path for res in results for s, d, t, path in res[3]})
    summary = measure.write(rows, hist, OUT, "time_pressure")
    print("time_pressure", {k: {m: round(x, 3) for m, x in v.items()} for k, v in summary.items()})
