"""Warm-started evolution in dangerous worlds (locked simple-12x12 random-mismatch setup).

World A "walls hurt": health HEALTH, each new collision (entering contact) costs 1;
at 0 the run ends without the goal. World B: walls hurt + hidden light (light
sensors read 0 beyond LIGHT_RANGE units of the goal, still blocked by walls).
Fitness stays outcome-only (path progress + goal - stuck penalty): a robot that
dies keeps the path progress it had made and gets no goal bonus.

For each GA winner (seeds 3, 19 = left-turners; 2, 13 = right-turners): a
population of 30 = the winner + 29 mutated copies, evolved 30 generations on
shared random 12x12 mazes (8 per generation, as in the locked GA). Final
controller = best on 20 validation mazes (seed 777) *in its own world*, among
the final population and the best-so-far individual.
Usage: python evolve_danger.py REPO GA_GENOMES_NPZ OUT_DIR [GENERATIONS]
"""

import csv
import sys
import time
from multiprocessing import Pool

REPO, GA_GENOMES, OUT = sys.argv[1:4]
GENS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from evolution.experiment import (  # noqa: E402
    TRAINING_NOISE_SEED_BASE, RandomMazeWorld, evaluate_course_set_outcome, sample_random_mazes)
from evolution.fitness import evaluate_outcome_fitness  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from evolution.metrics import record_run  # noqa: E402
from evolution.experiment import course_step_limit  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE  # noqa: E402

HEALTH = 15
LIGHT_RANGE = 6.0      # 1.5 maze cells of 4 units
POP = 30
STEP_CAP = 10000
NOISE = 0.05
WORLDS = {
    "baseline": ({}, {}),
    "A_walls_hurt": ({"health": HEALTH}, {}),
    "B_hurt_hidden_light": ({"health": HEALTH}, {"light_visible_distance": LIGHT_RANGE}),
}
ROBOTS = (3, 19, 2, 13)


def world_settings(name):
    world_options, vehicle_options = WORLDS[name]
    return world_options, {**DEFAULT_ROBOT_NOISE, **vehicle_options}


def new_ga(seed):
    return GeneticAlgorithm(population_size=POP, hidden_size=4, mutation_rate=0.15, mutation_scale=0.2,
                            seed=seed, fitness_function=evaluate_course_set_outcome, network_only=True,
                            light_occlusion="blocked")


def episode_score(network, maze, start, heading, noise_seed, world_name):
    world_options, robot_noise = world_settings(world_name)
    record = record_run(network, maze, start, heading,
                        max_steps=min(STEP_CAP, course_step_limit(maze, start=start)),
                        noise_seed=noise_seed, noise_stdev=NOISE, robot_noise=robot_noise,
                        light_occlusion="blocked", world_options=world_options)
    result = evaluate_outcome_fitness(record.world)
    return result["path_progress"] + float(result["reached_goal"])


def validation_score(ga, genome, world_name):
    network = ga.decode_genome(genome)
    mazes = sample_random_mazes(np.random.default_rng(777), 20)
    return float(np.mean([episode_score(network, rm.maze, rm.start, rm.start_heading, 800000 + i, world_name)
                          for i, rm in enumerate(mazes)]))


def run(args):
    robot, world_name = args
    world_options, robot_noise = world_settings(world_name)
    run_id = robot * 10 + list(WORLDS).index(world_name)
    ga = new_ga(1000 + run_id)
    original = np.load(GA_GENOMES)[f"seed{robot}"]
    ga.population[0].genome = original.copy()
    for individual in ga.population[1:]:
        individual.genome = ga.mutate(original)
    noise_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, 5000 + run_id])
    maze_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, 5000 + run_id, 1])
    log = open(f"{OUT}/progress_seed{robot}_{world_name}.txt", "w")
    start = time.time()
    conditions = {}

    def factory(controller):
        if ga.generation not in conditions:
            conditions.clear()
            conditions[ga.generation] = (sample_random_mazes(maze_rng), int(noise_rng.integers(2**31)))
        mazes, noise_seed_base = conditions[ga.generation]
        world = RandomMazeWorld(controller, mazes=mazes, noise_stdev=NOISE, noise_seed_base=noise_seed_base,
                                light_occlusion="blocked", robot_noise=robot_noise)
        for _, w in world.worlds:                 # turn on "walls hurt"
            w.health = world_options.get("health")
        return world

    for g in range(GENS):
        best = ga.evaluate_population(world_factory=factory, steps=STEP_CAP)
        h = ga.history[-1]
        courses = [ind.metrics["courses"] for ind in ga.population]
        h["gen_best_goal_rate"] = float(np.mean([c["reached_goal"] for c in best.metrics["courses"]]))
        h["mean_goal_rate"] = float(np.mean([[c["reached_goal"] for c in cs] for cs in courses]))
        h["mean_collisions_per_run"] = float(np.mean([[c["collisions"] for c in cs] for cs in courses]))
        print(f"gen {g} best {h['generation_best_fitness']:.3f} (goal {h['gen_best_goal_rate']:.2f}) "
              f"mean {h['mean_fitness']:.3f} pop goal rate {h['mean_goal_rate']:.2f} t={time.time() - start:.0f}s",
              file=log, flush=True)
        if g < GENS - 1:
            ga.next_generation()
    candidates = [ind.genome for ind in ga.population] + [ga.best_individual.genome]
    scores = [validation_score(ga, genome, world_name) for genome in candidates]
    chosen = candidates[int(np.argmax(scores))]
    print(f"validation: best {max(scores):.3f}; original winner "
          f"{validation_score(ga, original, world_name):.3f}", file=log, flush=True)
    return robot, world_name, chosen, [{"robot": robot, "world": world_name, **h} for h in ga.history]


if __name__ == "__main__":
    jobs = [(robot, world) for robot in ROBOTS for world in ("A_walls_hurt", "B_hurt_hidden_light")]
    with Pool(len(jobs)) as pool:
        results = pool.map(run, jobs)
    np.savez(f"{OUT}/evolved_genomes.npz", **{f"seed{r}_{w}": g for r, w, g, _ in results})
    history = [h for *_, hs in results for h in hs]
    with open(f"{OUT}/history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    print("done")
