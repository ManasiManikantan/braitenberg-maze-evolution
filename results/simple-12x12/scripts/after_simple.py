"""SIMPLE SETUP: network-only evolution on shared random 12x12 mazes.

Mirrors evolution/experiment.py run_experiment defaults: each generation the
whole population is evaluated on the same 8 random 12x12 loop-free mazes and
noise seed; per-maze limit 8 x shortest route at 0.05 units/step; outcome-only
fitness with the stuck penalty; 3 distance sensors (0, +/-60 deg) + 2 light
sensors (walls block); robot noise + heading noise 0.05; physical motor
convention. Mutation 0.15/0.2, hidden size 4.

Final controller per seed = best mean (progress + goal) on 20 fixed validation
mazes (seed 777), among the final population and the best-so-far individual.
Tests: 50 random 12x12 test mazes (seed 2026) and courses A/B/C x 10 trials,
for the evolved controllers and untrained networks.
Usage: python after_simple.py REPO OUT_DIR SEEDS GENERATIONS POP
"""

import csv
import sys
import time
from multiprocessing import Pool

REPO, OUT = sys.argv[1:3]
SEEDS, GENS, POP = map(int, sys.argv[3:6])
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.experiment import (  # noqa: E402
    TRAINING_NOISE_SEED_BASE, RandomMazeWorld, course_step_limit, evaluate_course_set_outcome,
    make_evaluation_courses, sample_random_mazes)
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
import measure  # noqa: E402

LIGHT = "blocked"
NOISE = 0.05
STEP_CAP = 10000
VALIDATION_SEED = 777
TEST_SEED = 2026
N_VALIDATION = 20
N_TEST = 50
COURSE_TRIALS = 10


def new_ga(seed, population_size=POP):
    return GeneticAlgorithm(population_size=population_size, hidden_size=4, mutation_rate=0.15,
                            mutation_scale=0.2, seed=seed,
                            fitness_function=evaluate_course_set_outcome, network_only=True,
                            light_occlusion=LIGHT)


def episode(controller, maze, start, heading, noise_seed, record=None):
    vehicle = Vehicle(controller=controller, noise_stdev=NOISE, noise_seed=noise_seed,
                      navigation_heuristics=False, start_position=start, start_heading=heading,
                      light_occlusion=LIGHT, **DEFAULT_ROBOT_NOISE)
    world = World(maze=maze, vehicle=vehicle)
    limit = min(STEP_CAP, course_step_limit(maze, start=start))
    flags = []
    while world.steps < limit and not world.reached_goal:
        before = world.collisions
        world.step()
        flags.append(world.collisions > before)
        if record is not None and world.steps % 5 == 0:
            record.append(vehicle.controller_inputs())
    row = {**measure.measure_world(world), **measure.contact_stats(flags)}
    row["limit"] = limit
    return row


def test_entries(validation=False):
    if validation:
        mazes = sample_random_mazes(np.random.default_rng(VALIDATION_SEED), N_VALIDATION)
        return [("validation", f"v{i}", rm.maze, rm.start, rm.start_heading, 800000 + i)
                for i, rm in enumerate(mazes)]
    out = [("random_12x12", f"r{i}", rm.maze, rm.start, rm.start_heading, 700000 + i)
           for i, rm in enumerate(sample_random_mazes(np.random.default_rng(TEST_SEED), N_TEST))]
    for name, _, maze, start, heading in make_evaluation_courses():
        out += [(name, f"t{t}", maze, start, heading, 500000 + t) for t in range(COURSE_TRIALS)]
    return out


def evaluate_set(genome, seed, label, validation=False):
    controller = new_ga(0, 4).decode_genome(genome)
    rows, inputs = [], []
    for set_name, maze_label, maze, start, heading, noise_seed in test_entries(validation):
        row = episode(controller, maze, start, heading, noise_seed,
                      record=inputs if (set_name == "random_12x12" and len(inputs) < 4000) else None)
        rows.append({"controller": label, "seed": seed, "set": set_name, "maze": maze_label, **row})
    return rows, np.array(inputs)


def validation_score(genome, seed):
    rows, _ = evaluate_set(genome, seed, "val", validation=True)
    return float(np.mean([r["path_progress"] + r["goal_success"] for r in rows]))


def run_seed(seed):
    ga = new_ga(seed)
    noise_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, seed])
    maze_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, seed, 1])
    log = open(f"{OUT}/progress_seed{seed}.txt", "w")
    start = time.time()
    conditions = {}

    def factory(controller):
        if ga.generation not in conditions:
            conditions.clear()
            conditions[ga.generation] = (sample_random_mazes(maze_rng), int(noise_rng.integers(2**31)))
        mazes, noise_seed_base = conditions[ga.generation]
        return RandomMazeWorld(controller, mazes=mazes, noise_stdev=NOISE,
                               noise_seed_base=noise_seed_base, light_occlusion=LIGHT)

    for g in range(GENS):
        best = ga.evaluate_population(world_factory=factory, steps=STEP_CAP)
        h = ga.history[-1]
        h["gen_best_goal_rate"] = float(np.mean([c["reached_goal"] for c in best.metrics["courses"]]))
        h["mean_goal_rate"] = float(np.mean([
            np.mean([c["reached_goal"] for c in ind.metrics["courses"]]) for ind in ga.population]))
        print(f"gen {g} best {h['generation_best_fitness']:.3f} (goal {h['gen_best_goal_rate']:.2f}) "
              f"mean {h['mean_fitness']:.3f} pop goal rate {h['mean_goal_rate']:.2f} "
              f"t={time.time() - start:.0f}s", file=log, flush=True)
        if g < GENS - 1:
            ga.next_generation()

    candidates = [ind.genome for ind in ga.population] + [ga.best_individual.genome]
    scores = [validation_score(genome, seed) for genome in candidates]
    chosen = candidates[int(np.argmax(scores))]
    print(f"validation: best {max(scores):.3f} (best-so-far individual {scores[-1]:.3f})",
          file=log, flush=True)
    return [{"seed": seed, **h} for h in ga.history], chosen


def test_job(args):
    genome, seed, label = args
    return evaluate_set(genome, seed, label)


if __name__ == "__main__":
    with Pool(min(SEEDS, 10)) as p:
        trained = p.map(run_seed, range(SEEDS))
    hist = [h for res in trained for h in res[0]]
    genomes = {f"seed{s}": res[1] for s, res in enumerate(trained)}
    initial = {f"seed{s}": new_ga(s).population[0].genome for s in range(SEEDS)}
    np.savez(f"{OUT}/evolved_genomes.npz", **genomes)
    np.savez(f"{OUT}/untrained_genomes.npz", **initial)
    jobs = [(genomes[k], int(k[4:]), "evolved") for k in genomes]
    jobs += [(initial[k], int(k[4:]), "untrained") for k in initial]
    with Pool(10) as p:
        results = p.map(test_job, jobs)
    rows = [r for chunk, _ in results for r in chunk]
    np.savez_compressed(f"{OUT}/visited_inputs.npz",
                        **{k: inputs for k, (_, inputs) in zip(genomes, results[:len(genomes)])})
    with open(f"{OUT}/test_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{OUT}/history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    print("done", len(rows), "episodes")
