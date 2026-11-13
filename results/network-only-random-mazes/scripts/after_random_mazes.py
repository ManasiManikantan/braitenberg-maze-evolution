"""RANDOM MAZES: network-only evolution on fresh random mazes, 60-degree side sensors.

Training mirrors evolution/experiment.py run_experiment defaults (random-maze
training, 3 mazes per evaluation, per-maze 2x time limits, outcome-only
fitness, robot noise, heading noise 0.05, mutation 0.15/0.2). Final controller
per seed = best mean outcome score on a fixed 20-maze validation set, chosen
among the final population and the best-so-far individual.

Tests: 50 random test mazes (separate seed) + courses 1-4 x 5 trials.
Also tests the time-pressure controllers (parallel sensors) on the same mazes.
Usage: python after_random_mazes.py REPO OUT_DIR TP_GENOMES SEEDS GENERATIONS POP
"""

import math
import sys
import time
from multiprocessing import Pool

REPO, OUT, TP_GENOMES = sys.argv[1:4]
SEEDS, GENS, POP = map(int, sys.argv[4:7])
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.experiment import (  # noqa: E402
    TRAINING_NOISE_SEED_BASE, RandomMazeWorld, course_step_limit, evaluate_course_set_outcome)
from evolution.fitness import evaluate_outcome_fitness  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.maze_generator import generate_perfect_maze  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, DEFAULT_SIDE_SENSOR_ANGLE, Vehicle  # noqa: E402
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


def random_mazes(seed, count):
    rng = np.random.default_rng(seed)
    return [generate_perfect_maze(rng) for _ in range(count)]


def episode(controller, maze, start, heading, noise_seed, side_angle, keep=False):
    """Run one episode; return outcome/contact/wall-following stats (and the path)."""
    vehicle = Vehicle(controller=controller, noise_stdev=NOISE, noise_seed=noise_seed,
                      navigation_heuristics=False, start_position=start, start_heading=heading,
                      side_sensor_angle=side_angle, **DEFAULT_ROBOT_NOISE)
    world = World(maze=maze, vehicle=vehicle)
    limit = min(STEP_CAP, course_step_limit(maze, start=start))
    flags, wall_side = [], []
    x0, y0, h0 = vehicle.x_pos, vehicle.y_pos, vehicle.orientation
    while world.steps < limit and not world.reached_goal:
        before = world.collisions
        world.step()
        hit = world.collisions > before
        flags.append(hit)
        if not hit:
            # wall beside the body? cast perpendicular rays from the centre
            h = vehicle.orientation
            left = maze.raycast((vehicle.x_pos, vehicle.y_pos), (-math.sin(h), math.cos(h)), 10.0)
            right = maze.raycast((vehicle.x_pos, vehicle.y_pos), (math.sin(h), -math.cos(h)), 10.0)
            near = min(left, right) - vehicle.radius
            wall_side.append(("L" if left < right else "R") if near < 1.0 else "-")
    row = {**measure.measure_world(world), **measure.contact_stats(flags)}
    moving = len(wall_side)
    beside = [s for s in wall_side if s != "-"]
    row["pct_moving_with_wall_beside"] = 100.0 * len(beside) / max(moving, 1)
    row["wall_side_left_share"] = (sum(s == "L" for s in beside) / len(beside)) if beside else float("nan")
    path = np.array(world.position_history)
    # early trajectory in the robot's own start frame (fixed-arc test)
    rel = path[:EARLY_STEPS + 1] - (x0, y0)
    c, s = math.cos(-h0), math.sin(-h0)
    row["_early"] = np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1)
    if keep:
        row["_path"] = path
    return row


def evaluate_controller_set(genome, seed, side_angle, label, validation=False):
    ga = GeneticAlgorithm(population_size=4, hidden_size=4, seed=0)
    controller = ga.decode_genome(genome)
    rows = []
    mazes = random_mazes(VALIDATION_SEED if validation else TEST_SEED,
                         N_VALIDATION if validation else N_TEST)
    for index, rm in enumerate(mazes):
        row = episode(controller, rm.maze, rm.start, rm.start_heading,
                      seed * 100000 + index + 700000, side_angle)
        rows.append({"controller": label, "seed": seed, "set": "random_test", "course": f"r{index}",
                     "trial": 0, **row})
    if validation:
        return rows
    for d in (1, 2, 3, 4):
        maze = Maze(difficulty=d)
        for trial in range(COURSE_TRIALS):
            row = episode(controller, maze, (0.0, 0.0), 0.0,
                          seed * 1000 + d * 10 + trial + 500000, side_angle)
            rows.append({"controller": label, "seed": seed, "set": "courses", "course": str(d),
                         "trial": trial, **row})
    return rows


def validation_score(genome, seed):
    rows = evaluate_controller_set(genome, seed, DEFAULT_SIDE_SENSOR_ANGLE, "val", validation=True)
    return float(np.mean([r["path_progress"] + r["goal_success"] for r in rows]))


def run_seed(seed):
    ga = GeneticAlgorithm(population_size=POP, hidden_size=4, mutation_rate=0.15,
                          mutation_scale=0.2, seed=seed,
                          fitness_function=evaluate_course_set_outcome, network_only=True)
    noise_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, seed])
    maze_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, seed, 1])
    log = open(f"{OUT}/random_mazes_progress_seed{seed}.txt", "w")
    start = time.time()

    def factory(controller):
        return RandomMazeWorld(controller, maze_rng, noise_stdev=NOISE,
                               noise_seed_base=int(noise_rng.integers(2**31)))

    for g in range(GENS):
        best = ga.evaluate_population(world_factory=factory, steps=STEP_CAP)
        h = ga.history[-1]
        goals = sum(c["reached_goal"] for c in best.metrics["courses"])
        h["gen_best_goals"] = goals
        print(f"gen {g} best {h['generation_best_fitness']:.3f} mean {h['mean_fitness']:.3f} "
              f"goals {goals}/3 t={time.time() - start:.0f}s", file=log, flush=True)
        if g < GENS - 1:
            ga.next_generation()

    candidates = [ind.genome for ind in ga.population] + [ga.best_individual.genome]
    scores = [validation_score(genome, seed) for genome in candidates]
    chosen = candidates[int(np.argmax(scores))]
    print(f"validation: best {max(scores):.3f} (best-so-far individual scored {scores[-1]:.3f})",
          file=log, flush=True)
    hist = [{"seed": seed, **h} for h in ga.history]
    return hist, chosen


def test_job(args):
    genome, seed, side_angle, label = args
    return evaluate_controller_set(genome, seed, side_angle, label)


def strip(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


if __name__ == "__main__":
    with Pool(min(SEEDS, 10)) as p:
        trained = p.map(run_seed, range(SEEDS))
    hist = [h for res in trained for h in res[0]]
    genomes = {f"seed{s}": res[1] for s, res in enumerate(trained)}
    np.savez(f"{OUT}/random_mazes_genomes.npz", **genomes)

    tp = np.load(TP_GENOMES)
    jobs = [(genomes[k], int(k[4:]), DEFAULT_SIDE_SENSOR_ANGLE, "random_mazes") for k in genomes]
    jobs += [(tp[k], int(k[4:]), 0.0, "time_pressure") for k in tp.files]
    with Pool(10) as p:
        results = p.map(test_job, jobs)
    rows = [r for chunk in results for r in chunk]

    early = {f"{r['controller']}_seed{r['seed']}_{r['set']}_{r['course']}_t{r['trial']}": r["_early"]
             for r in rows}
    np.savez_compressed(f"{OUT}/early_trajectories.npz", **early)
    flat = strip(rows)
    import csv
    with open(f"{OUT}/random_mazes_test_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    with open(f"{OUT}/random_mazes_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    print("done", len(flat), "episodes")
