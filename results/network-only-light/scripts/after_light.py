"""LIGHT: network-only evolution on random mazes with goal-light sensors + curriculum.

Training mirrors evolution/experiment.py run_experiment defaults (random-maze
training with DEFAULT_CURRICULUM, 3 mazes per evaluation, per-maze 2x time
limits, outcome-only fitness, robot noise incl. light noise, heading noise 0.05,
60-degree side sensors, mutation 0.15/0.2) in the given light occlusion mode.
Final controller per seed = best mean (progress + goal) on a fixed 20-maze
full-size validation set, chosen among the final population and the
best-so-far individual.

Tests: the same 50 random test mazes (seed 2026) and courses 1-4 x 5 trials as
the previous random-maze run, for the evolved controllers and for untrained
5-input networks (first genome of each seed's initial population).
Usage: python after_light.py REPO OUT_DIR LIGHT_MODE SEEDS GENERATIONS POP
"""

import csv
import math
import sys
import time
from multiprocessing import Pool

REPO, OUT, LIGHT = sys.argv[1:4]
SEEDS, GENS, POP = map(int, sys.argv[4:7])
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.experiment import (  # noqa: E402
    TRAINING_NOISE_SEED_BASE, RandomMazeWorld, course_step_limit, curriculum_maze_options,
    evaluate_course_set_outcome)
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.maze_generator import generate_perfect_maze  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
import measure  # noqa: E402

NOISE = 0.05
STEP_CAP = 3000
VALIDATION_SEED = 777
TEST_SEED = 2026
N_VALIDATION = 20
N_TEST = 50
COURSE_TRIALS = 5
EARLY_STEPS = 300


def new_ga(seed, population_size=POP):
    return GeneticAlgorithm(population_size=population_size, hidden_size=4, mutation_rate=0.15,
                            mutation_scale=0.2, seed=seed,
                            fitness_function=evaluate_course_set_outcome, network_only=True,
                            light_occlusion=LIGHT)


def random_mazes(seed, count):
    rng = np.random.default_rng(seed)
    return [generate_perfect_maze(rng) for _ in range(count)]


def episode(controller, maze, start, heading, noise_seed, record=None):
    vehicle = Vehicle(controller=controller, noise_stdev=NOISE, noise_seed=noise_seed,
                      navigation_heuristics=False, start_position=start, start_heading=heading,
                      light_occlusion=LIGHT, **DEFAULT_ROBOT_NOISE)
    world = World(maze=maze, vehicle=vehicle)
    limit = min(STEP_CAP, course_step_limit(maze, start=start))
    flags, wall_side = [], []
    x0, y0, h0 = vehicle.x_pos, vehicle.y_pos, vehicle.orientation
    while world.steps < limit and not world.reached_goal:
        before = world.collisions
        world.step()
        hit = world.collisions > before
        flags.append(hit)
        if record is not None and world.steps % 5 == 0:
            record.append(vehicle.controller_inputs())
        if not hit:
            h = vehicle.orientation
            left = maze.raycast((vehicle.x_pos, vehicle.y_pos), (-math.sin(h), math.cos(h)), 10.0)
            right = maze.raycast((vehicle.x_pos, vehicle.y_pos), (math.sin(h), -math.cos(h)), 10.0)
            near = min(left, right) - vehicle.radius
            wall_side.append(("L" if left < right else "R") if near < 1.0 else "-")
    row = {**measure.measure_world(world), **measure.contact_stats(flags)}
    beside = [s for s in wall_side if s != "-"]
    row["pct_moving_with_wall_beside"] = 100.0 * len(beside) / max(len(wall_side), 1)
    row["wall_side_left_share"] = (sum(s == "L" for s in beside) / len(beside)) if beside else float("nan")
    path = np.array(world.position_history)
    rel = path[:EARLY_STEPS + 1] - (x0, y0)
    c, s = math.cos(-h0), math.sin(-h0)
    row["_early"] = np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1)
    return row


def evaluate_controller_set(genome, seed, label, validation=False):
    controller = new_ga(0, 4).decode_genome(genome)
    rows, inputs = [], []
    mazes = random_mazes(VALIDATION_SEED if validation else TEST_SEED,
                         N_VALIDATION if validation else N_TEST)
    for index, rm in enumerate(mazes):
        row = episode(controller, rm.maze, rm.start, rm.start_heading,
                      seed * 100000 + index + 700000,
                      record=inputs if (not validation and index < 20) else None)
        rows.append({"controller": label, "seed": seed, "set": "random_test", "course": f"r{index}",
                     "trial": 0, **row})
    if not validation:
        for d in (1, 2, 3, 4):
            maze = Maze(difficulty=d)
            for trial in range(COURSE_TRIALS):
                row = episode(controller, maze, (0.0, 0.0), 0.0, seed * 1000 + d * 10 + trial + 500000)
                rows.append({"controller": label, "seed": seed, "set": "courses", "course": str(d),
                             "trial": trial, **row})
    return rows, np.array(inputs)


def validation_score(genome, seed):
    rows, _ = evaluate_controller_set(genome, seed, "val", validation=True)
    return float(np.mean([r["path_progress"] + r["goal_success"] for r in rows]))


def run_seed(seed):
    ga = new_ga(seed)
    noise_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, seed])
    maze_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, seed, 1])
    log = open(f"{OUT}/light_{LIGHT}_progress_seed{seed}.txt", "w")
    start = time.time()

    def factory(controller):
        return RandomMazeWorld(controller, maze_rng, noise_stdev=NOISE,
                               noise_seed_base=int(noise_rng.integers(2**31)),
                               light_occlusion=LIGHT,
                               maze_options=curriculum_maze_options(ga.generation, GENS))

    for g in range(GENS):
        best = ga.evaluate_population(world_factory=factory, steps=STEP_CAP)
        h = ga.history[-1]
        h["maze_size"] = curriculum_maze_options(g, GENS)["size"]
        h["gen_best_goals"] = sum(c["reached_goal"] for c in best.metrics["courses"])
        h["mean_goal_rate"] = float(np.mean([
            np.mean([c["reached_goal"] for c in ind.metrics["courses"]]) for ind in ga.population
        ]))
        h["mean_path_progress"] = float(np.mean([
            ind.metrics["score_components"]["path_progress"] for ind in ga.population
        ]))
        print(f"gen {g} size {h['maze_size']:.0f} best {h['generation_best_fitness']:.3f} "
              f"mean {h['mean_fitness']:.3f} pop goal rate {h['mean_goal_rate']:.2f} "
              f"t={time.time() - start:.0f}s", file=log, flush=True)
        if g < GENS - 1:
            ga.next_generation()

    candidates = [ind.genome for ind in ga.population] + [ga.best_individual.genome]
    scores = [validation_score(genome, seed) for genome in candidates]
    chosen = candidates[int(np.argmax(scores))]
    print(f"validation: best {max(scores):.3f} (best-so-far individual scored {scores[-1]:.3f})",
          file=log, flush=True)
    return [{"seed": seed, **h} for h in ga.history], chosen


def test_job(args):
    genome, seed, label = args
    return evaluate_controller_set(genome, seed, label)


if __name__ == "__main__":
    with Pool(min(SEEDS, 10)) as p:
        trained = p.map(run_seed, range(SEEDS))
    hist = [h for res in trained for h in res[0]]
    genomes = {f"seed{s}": res[1] for s, res in enumerate(trained)}
    initial = {f"seed{s}": new_ga(s).population[0].genome for s in range(SEEDS)}
    np.savez(f"{OUT}/light_{LIGHT}_genomes.npz", **genomes)
    np.savez(f"{OUT}/untrained_{LIGHT}_genomes.npz", **initial)

    jobs = [(genomes[k], int(k[4:]), f"light_{LIGHT}") for k in genomes]
    jobs += [(initial[k], int(k[4:]), f"untrained_{LIGHT}") for k in initial]
    with Pool(10) as p:
        results = p.map(test_job, jobs)
    rows = [r for chunk, _ in results for r in chunk]
    np.savez_compressed(f"{OUT}/light_{LIGHT}_visited_inputs.npz",
                        **{k: inputs for k, (_, inputs) in zip(genomes, results[:len(genomes)])})
    np.savez_compressed(
        f"{OUT}/light_{LIGHT}_early_trajectories.npz",
        **{f"{r['controller']}_seed{r['seed']}_{r['set']}_{r['course']}_t{r['trial']}": r["_early"]
           for r in rows})
    flat = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(f"{OUT}/light_{LIGHT}_test_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    with open(f"{OUT}/light_{LIGHT}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    print("done", len(flat), "episodes")
