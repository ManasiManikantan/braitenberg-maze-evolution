"""Walls-hurt v2 (no scraping loophole): copy of evolve_zoo.py with only these runs.

walls_hurt_v2: health 15, 1 per new hit + CONTACT_STEP_COST per step in contact.

Original description: Personality zoo: warm-started evolution in more worlds (locked simple-12x12 random-mismatch setup).

Worlds (all otherwise identical to the locked setup, fitness outcome-only):
- safe: the plain locked world (control for "just more generations").
- battery: energy budget per maze = BATTERY_PER_UNIT x shortest-route length; each step costs
  |left motor| + |right motor|; the run ends when empty (keeps its path progress, no goal).
- explorer: hidden light (sensors read 0 beyond 6 units) + the goal moved to a random dead end
  of each maze.
- walls_hurt: health 15, each new collision costs 1 (as in the danger runs).

Runs (population 30 = parent + 29 mutated copies, 20 generations, all in parallel):
seeds 3 and 2 in safe, battery, explorer; the hand-set fear robot twice in safe and twice in
walls_hurt. The generation-best genome is saved every generation. Final controller = best on
20 validation mazes (seed 777) in its own world, among the final population and best-so-far.
Usage: python evolve_zoo.py REPO GA_GENOMES_NPZ FEAR_NPZ OUT_DIR [GENERATIONS]
"""

import csv
import sys
import time
from multiprocessing import Pool

REPO, GA_GENOMES, FEAR, OUT = sys.argv[1:5]
GENS = int(sys.argv[5]) if len(sys.argv) > 5 else 20
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from evolution.experiment import (  # noqa: E402
    TRAINING_NOISE_SEED_BASE, RandomMazeWorld, course_step_limit, evaluate_course_set_outcome,
    sample_random_mazes)
from evolution.fitness import evaluate_outcome_fitness, path_distances  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from evolution.metrics import record_run  # noqa: E402
from sim.maze_generator import with_goal_in_dead_end  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE  # noqa: E402

POP = 30
STEP_CAP = 10000
NOISE = 0.05
HEALTH = 15
LIGHT_RANGE = 6.0
BATTERY_PER_UNIT = 100.0
CONTACT_STEP_COST = 0.1
ROBOT_RADIUS = 1.0
RUNS = [("seed3", "walls_hurt_v2", 0), ("seed2", "walls_hurt_v2", 0),
        ("fear", "walls_hurt_v2", 0), ("fear", "walls_hurt_v2", 1)]


def prepare_mazes(world, mazes, rng):
    return [with_goal_in_dead_end(rm, rng) for rm in mazes] if world == "explorer" else list(mazes)


def options(world, random_maze):
    """(world_options, robot_noise) for one maze in ``world``."""
    world_options, vehicle = {}, {}
    if world == "walls_hurt":
        world_options["health"] = HEALTH
    if world == "walls_hurt_v2":
        world_options["health"] = HEALTH
        world_options["contact_step_cost"] = CONTACT_STEP_COST
    if world == "battery":
        route = float(path_distances(random_maze.maze, [random_maze.start], ROBOT_RADIUS)[0])
        world_options["energy"] = BATTERY_PER_UNIT * route
    if world == "explorer":
        vehicle["light_visible_distance"] = LIGHT_RANGE
    return world_options, {**DEFAULT_ROBOT_NOISE, **vehicle}


def new_ga(seed):
    return GeneticAlgorithm(population_size=POP, hidden_size=4, mutation_rate=0.15, mutation_scale=0.2,
                            seed=seed, fitness_function=evaluate_course_set_outcome, network_only=True,
                            light_occlusion="blocked")


def parent_genome(name):
    return np.load(FEAR)["fear"] if name == "fear" else np.load(GA_GENOMES)[name]


def validation_score(ga, genome, world):
    network = ga.decode_genome(genome)
    rng = np.random.default_rng([777, 1])
    mazes = prepare_mazes(world, sample_random_mazes(np.random.default_rng(777), 20), rng)
    total = []
    for i, rm in enumerate(mazes):
        world_options, robot_noise = options(world, rm)
        record = record_run(network, rm.maze, rm.start, rm.start_heading,
                            max_steps=min(STEP_CAP, course_step_limit(rm.maze, start=rm.start)),
                            noise_seed=800000 + i, noise_stdev=NOISE, robot_noise=robot_noise,
                            light_occlusion="blocked", world_options=world_options)
        result = evaluate_outcome_fitness(record.world)
        total.append(result["path_progress"] + float(result["reached_goal"]))
    return float(np.mean(total))


def run(args):
    parent, world, replicate = args
    run_id = RUNS.index(args)
    tag = f"{parent}_{world}_{replicate}"
    ga = new_ga(3000 + run_id)
    original = parent_genome(parent)
    ga.population[0].genome = original.copy()
    for individual in ga.population[1:]:
        individual.genome = ga.mutate(original)
    noise_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, 8000 + run_id])
    maze_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, 8000 + run_id, 1])
    log = open(f"{OUT}/progress_{tag}.txt", "w")
    start = time.time()
    conditions = {}

    def factory(controller):
        if ga.generation not in conditions:
            conditions.clear()
            mazes = prepare_mazes(world, sample_random_mazes(maze_rng), maze_rng)
            conditions[ga.generation] = (mazes, int(noise_rng.integers(2**31)))
        mazes, noise_seed_base = conditions[ga.generation]
        robot_noise = options(world, mazes[0])[1]
        multi = RandomMazeWorld(controller, mazes=mazes, noise_stdev=NOISE, noise_seed_base=noise_seed_base,
                                light_occlusion="blocked", robot_noise=robot_noise)
        for (_, w), rm in zip(multi.worlds, mazes):
            world_options = options(world, rm)[0]
            w.health = world_options.get("health")
            w.energy = world_options.get("energy")
            w.contact_step_cost = world_options.get("contact_step_cost", 0.0)
        return multi

    best_per_generation = []
    for g in range(GENS):
        best = ga.evaluate_population(world_factory=factory, steps=STEP_CAP)
        best_per_generation.append(best.genome.copy())
        h = ga.history[-1]
        h["gen_best_goal_rate"] = float(np.mean([c["reached_goal"] for c in best.metrics["courses"]]))
        h["mean_goal_rate"] = float(np.mean([[c["reached_goal"] for c in ind.metrics["courses"]]
                                             for ind in ga.population]))
        print(f"gen {g} best {h['generation_best_fitness']:.3f} (goal {h['gen_best_goal_rate']:.2f}) "
              f"mean {h['mean_fitness']:.3f} pop goal rate {h['mean_goal_rate']:.2f} t={time.time() - start:.0f}s",
              file=log, flush=True)
        if g < GENS - 1:
            ga.next_generation()
    candidates = [ind.genome for ind in ga.population] + [ga.best_individual.genome]
    scores = [validation_score(ga, genome, world) for genome in candidates]
    chosen = candidates[int(np.argmax(scores))]
    print(f"validation: best {max(scores):.3f}; parent {validation_score(ga, original, world):.3f}",
          file=log, flush=True)
    return tag, chosen, np.array(best_per_generation), [{"run": tag, **h} for h in ga.history]


if __name__ == "__main__":
    with Pool(len(RUNS)) as pool:
        results = pool.map(run, RUNS)
    np.savez(f"{OUT}/evolved_genomes.npz", **{tag: g for tag, g, _, _ in results})
    np.savez(f"{OUT}/best_per_generation.npz", **{tag: b for tag, _, b, _ in results})
    history = [h for *_, hs in results for h in hs]
    with open(f"{OUT}/history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    print("done")
