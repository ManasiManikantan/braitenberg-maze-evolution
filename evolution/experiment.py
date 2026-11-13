"""Reproducible baseline, evolution, transfer, and heuristic-ablation study."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from evolution.fitness import evaluate_fitness, evaluate_outcome_fitness, path_distances
from evolution.genetic_algorithm import GeneticAlgorithm
from sim.maze import Maze
from sim.maze_generator import generate_perfect_maze
from sim.held_out_courses import TEST_COURSE_NAMES, make_test_course
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle
from sim.world import World

TRAINING_COURSES = (1, 2, 3, 4)
DEFAULT_SEEDS = 3
DEFAULT_GENERATIONS = 10
DEFAULT_POPULATION_SIZE = 30
DEFAULT_STEPS = 10000  # overall cap; per-maze time limits normally end runs first
DEFAULT_NOISE_STDEV = 0.05
TRAINING_NOISE_SEED_BASE = 900000
SUCCESS_BONUS = 100.0
SPEED_BONUS = 50.0
COLLISION_SCORE_FLOOR = -100.0
# Per-course time limit = factor * shortest-path length / reference speed.
# 0.05 maze units/step is ~75% of top speed (1/15); evolved network-only
# controllers averaged 0.052 on steps when they were not blocked by a wall.
# A factor of 0 (or None) disables per-course limits ("no time pressure").
# 8x leaves room for strategies that explore dead ends (a wall follower's
# route is typically 2-5x the shortest route).
DEFAULT_TIME_LIMIT_FACTOR = 8.0
DEFAULT_REFERENCE_SPEED = 0.05
# Network-only training uses random 12 x 12 loop-free mazes by default: each
# generation samples DEFAULT_MAZES_PER_EVALUATION mazes shared by the whole
# population. Test courses A/B/C (sim.held_out_courses) are never trained on.
DEFAULT_TRAINING = "random_mazes"
DEFAULT_MAZES_PER_EVALUATION = 8
DEFAULT_MAZE_OPTIONS = {"size": 12.0, "min_route_cells": 2, "max_route_cells": 8}
DEFAULT_MIRROR_PROBABILITY = 0.0
# Goal-light sensors for network-only evolution: "blocked" (walls block the
# light), "through" (they do not), or None (3-input network, no light).
DEFAULT_LIGHT_OCCLUSION = "blocked"
# Optional random-maze curriculum (off by default; pass curriculum= to use it):
# (fraction of the run at which the stage starts, generate_perfect_maze options).
DEFAULT_CURRICULUM = (
    (0.0, {"size": 12.0, "min_route_cells": 2, "max_route_cells": 3}),
    (1 / 3, {"size": 16.0, "min_route_cells": 2, "max_route_cells": 5}),
    (2 / 3, {"size": 20.0, "min_route_cells": 3, "max_route_cells": 6}),
)


def curriculum_maze_options(generation, generations, curriculum=DEFAULT_CURRICULUM):
    """Return the maze options of the curriculum stage active at ``generation``.

    ``curriculum`` is a sequence of ``(start_fraction, options)`` sorted by
    start fraction, the first starting at 0. ``None`` or empty means the
    default full-size mazes for the whole run.
    """
    if not curriculum:
        return {}
    if curriculum[0][0] != 0:
        raise ValueError("the first curriculum stage must start at 0")
    progress = generation / max(generations, 1)
    options = curriculum[0][1]
    for start_fraction, stage_options in curriculum:
        if progress >= start_fraction - 1e-12:
            options = stage_options
    return dict(options)
HISTORY_FIELDS = (
    "training_seed",
    "generation",
    "best_fitness",
    "generation_best_fitness",
    "mean_fitness",
    "stagnant_generations",
    "mutation_rate",
)
RESULT_FIELDS = (
    "training_seed",
    "controller",
    "behavior_mode",
    "course",
    "course_group",
    "noise_seed",
    "noise_stdev",
    "goal_success",
    "collision_count",
    "completion_steps",
    "path_length",
    "fitness",
    "evaluation_steps",
)


def course_step_limit(maze, time_limit_factor=DEFAULT_TIME_LIMIT_FACTOR,
                      reference_speed=DEFAULT_REFERENCE_SPEED, radius=1.0, start=(0.0, 0.0)):
    """Return a course's step limit, or None when per-course limits are off.

    The limit is ``time_limit_factor`` times the steps needed to drive the
    shortest robot-sized path from ``start`` to the goal at
    ``reference_speed`` maze units per step.
    """
    if not time_limit_factor:
        return None
    if time_limit_factor < 0 or reference_speed <= 0:
        raise ValueError("time_limit_factor must be >= 0 and reference_speed positive")
    distance = float(path_distances(maze, [start], radius)[0])
    return int(math.ceil(time_limit_factor * distance / reference_speed))


def _episode_steps(steps, limit):
    return steps if limit is None else min(steps, limit)


class MultiCourseWorld:
    """Run one controller on each training course and expose mean fitness.

    ``network_only=True`` (the default) disables the hand-coded navigation
    heuristics so the controller alone drives the robot. ``robot_noise``
    defaults to ``DEFAULT_ROBOT_NOISE``; pass ``{}`` to disable it.
    Each course runs for at most ``course_step_limit`` steps (see
    ``time_limit_factor`` / ``reference_speed``), capped by ``run(steps)``.
    """

    def __init__(self, controller, noise_stdev=0.0, noise_seed_base=TRAINING_NOISE_SEED_BASE,
                 network_only=True, robot_noise=None,
                 time_limit_factor=DEFAULT_TIME_LIMIT_FACTOR,
                 reference_speed=DEFAULT_REFERENCE_SPEED, light_occlusion=None):
        robot_noise = DEFAULT_ROBOT_NOISE if robot_noise is None else robot_noise
        self.worlds = [
            (
                f"course_{difficulty}",
                World(
                    maze=Maze(difficulty=difficulty),
                    vehicle=Vehicle(
                        controller=controller,
                        noise_stdev=noise_stdev,
                        noise_seed=noise_seed_base + difficulty,
                        navigation_heuristics=not network_only,
                        light_occlusion=light_occlusion,
                        **robot_noise,
                    ),
                ),
            )
            for difficulty in TRAINING_COURSES
        ]
        self.step_limits = [
            course_step_limit(world.maze, time_limit_factor, reference_speed)
            for _, world in self.worlds
        ]

    def run(self, steps):
        self.steps = steps
        for (_, world), limit in zip(self.worlds, self.step_limits):
            world.run(_episode_steps(steps, limit))


def sample_random_mazes(maze_rng, count=DEFAULT_MAZES_PER_EVALUATION,
                        mirror_probability=DEFAULT_MIRROR_PROBABILITY, maze_options=None):
    """Generate ``count`` random perfect mazes (``RandomMaze`` objects)."""
    maze_options = DEFAULT_MAZE_OPTIONS if maze_options is None else maze_options
    return [
        generate_perfect_maze(maze_rng, mirror_probability=mirror_probability, **maze_options)
        for _ in range(count)
    ]


class RandomMazeWorld:
    """Run one controller on random mazes.

    Uses ``mazes`` (a list of ``RandomMaze``) when given, e.g. the same mazes
    for a whole generation; otherwise samples ``mazes_per_evaluation`` new
    perfect mazes from ``maze_rng`` with ``maze_options`` (default
    ``DEFAULT_MAZE_OPTIONS``, 12 x 12). Each maze starts at its random start
    cell and heading and runs for at most its own ``course_step_limit``
    measured from that start. Same interface as ``MultiCourseWorld``, so the
    course-set fitness functions apply.
    """

    def __init__(self, controller, maze_rng=None, mazes_per_evaluation=DEFAULT_MAZES_PER_EVALUATION,
                 mirror_probability=DEFAULT_MIRROR_PROBABILITY, noise_stdev=0.0,
                 noise_seed_base=TRAINING_NOISE_SEED_BASE, network_only=True, robot_noise=None,
                 time_limit_factor=DEFAULT_TIME_LIMIT_FACTOR,
                 reference_speed=DEFAULT_REFERENCE_SPEED, maze_options=None,
                 light_occlusion=None, mazes=None):
        robot_noise = DEFAULT_ROBOT_NOISE if robot_noise is None else robot_noise
        if mazes is None:
            if maze_rng is None:
                raise ValueError("give either mazes or maze_rng")
            if mazes_per_evaluation < 1:
                raise ValueError("mazes_per_evaluation must be at least 1")
            mazes = sample_random_mazes(maze_rng, mazes_per_evaluation, mirror_probability,
                                        maze_options)
        self.mazes = list(mazes)
        self.worlds = [
            (
                f"random_maze_{index}",
                World(
                    maze=random_maze.maze,
                    vehicle=Vehicle(
                        controller=controller,
                        noise_stdev=noise_stdev,
                        noise_seed=noise_seed_base + index,
                        navigation_heuristics=not network_only,
                        start_position=random_maze.start,
                        start_heading=random_maze.start_heading,
                        light_occlusion=light_occlusion,
                        **robot_noise,
                    ),
                ),
            )
            for index, random_maze in enumerate(self.mazes)
        ]
        self.step_limits = [
            course_step_limit(random_maze.maze, time_limit_factor, reference_speed,
                              start=random_maze.start)
            for random_maze in self.mazes
        ]

    def run(self, steps):
        self.steps = steps
        for (_, world), limit in zip(self.worlds, self.step_limits):
            world.run(_episode_steps(steps, limit))


def _training_score(result, world, steps):
    """Selection score: clipped base fitness plus a large reward for finishing fast."""
    score = max(result["score"], COLLISION_SCORE_FLOOR)
    if result["reached_goal"]:
        score += SUCCESS_BONUS + SPEED_BONUS * (1 - world.steps / max(steps, 1))
    return score


def evaluate_course_set(multi_course_world):
    steps = getattr(multi_course_world, "steps", 0)
    results = []
    for course_name, world in multi_course_world.worlds:
        result = evaluate_fitness(world)
        result["score"] = _training_score(result, world, steps)
        results.append((course_name, result))
    component_names = results[0][1]["score_components"]
    score_components = {
        component: sum(result["score_components"][component] for _, result in results)
        / len(results)
        for component in component_names
    }
    return {
        "score": sum(result["score"] for _, result in results) / len(results),
        "score_components": score_components,
        "courses": [
            {"course": course_name, **result}
            for course_name, result in results
        ],
    }


def evaluate_course_set_outcome(multi_course_world):
    """Mean outcome-only fitness across the training courses (network-only default)."""
    results = [
        (course_name, evaluate_outcome_fitness(world))
        for course_name, world in multi_course_world.worlds
    ]
    component_names = results[0][1]["score_components"]
    return {
        "score": sum(result["score"] for _, result in results) / len(results),
        "score_components": {
            component: sum(result["score_components"][component] for _, result in results)
            / len(results)
            for component in component_names
        },
        "courses": [
            {"course": course_name, **result}
            for course_name, result in results
        ],
    }


def make_evaluation_courses():
    """Return the held-out 12 x 12 test courses A, B, C.

    Entries are ``(name, group, maze, start_position, start_heading)``.
    """
    entries = []
    for name in TEST_COURSE_NAMES:
        maze, start, heading = make_test_course(name)
        entries.append((f"course_{name}", "test", maze, start, heading))
    return entries


def make_legacy_courses():
    """Return the old courses 1-4 and two old held-out layouts (kept, unused).

    Entries are ``(name, group, maze)``; the robot starts at the origin facing +x.
    """
    courses = [
        (f"course_{difficulty}", "training", Maze(difficulty=difficulty))
        for difficulty in TRAINING_COURSES
    ]
    courses.extend((
        (
            "heldout_gates",
            "held_out",
            Maze(
                walls=[
                    ((2.0, -10.0), (2.0, 2.0)),
                    ((5.0, 2.0), (5.0, 10.0)),
                ],
                obstacles=[
                    (-7.0, -7.0, -6.0, -6.0),
                    (0.0, -3.0, 1.0, -2.0),
                    (8.0, 0.0, 9.0, 1.0),
                ],
            ),
        ),
        (
            "heldout_baffles",
            "held_out",
            Maze(
                walls=[
                    ((2.0, -2.0), (10.0, -2.0)),
                    ((5.0, -10.0), (5.0, -5.5)),
                ],
                obstacles=[
                    (-7.0, 4.0, -6.0, 5.0),
                    (0.0, 3.0, 1.0, 4.0),
                    (8.0, 3.0, 9.0, 4.0),
                ],
            ),
        ),
    ))
    return courses


def evaluate_controller(
    controller,
    controller_name,
    behavior_mode,
    training_seed,
    courses,
    steps,
    noise_stdev,
    robot_noise=None,
    time_limit_factor=DEFAULT_TIME_LIMIT_FACTOR,
    reference_speed=DEFAULT_REFERENCE_SPEED,
    light_occlusion=None,
):
    """Evaluate one controller on every course with reproducible noise.

    ``courses`` entries are ``(name, group, maze)`` (start at the origin facing
    +x) or ``(name, group, maze, start_position, start_heading)``.

    ``robot_noise`` (default ``DEFAULT_ROBOT_NOISE``) and the per-course step
    limits apply to every controller and mode so comparisons share the same
    conditions.
    """
    robot_noise = DEFAULT_ROBOT_NOISE if robot_noise is None else robot_noise
    use_heuristics = behavior_mode == "full"
    records = []
    for course_index, entry in enumerate(courses):
        course_name, course_group, maze = entry[:3]
        start, heading = entry[3:5] if len(entry) >= 5 else ((0.0, 0.0), 0.0)
        noise_seed = training_seed * 10000 + course_index
        vehicle = Vehicle(
            controller=controller,
            noise_stdev=noise_stdev,
            noise_seed=noise_seed,
            navigation_heuristics=use_heuristics,
            light_occlusion=light_occlusion,
            start_position=start,
            start_heading=heading,
            **robot_noise,
        )
        world = World(maze=maze, vehicle=vehicle)
        world.run(_episode_steps(
            steps, course_step_limit(maze, time_limit_factor, reference_speed, start=start)
        ))
        metrics = evaluate_fitness(world)
        records.append({
            "training_seed": training_seed,
            "controller": controller_name,
            "behavior_mode": behavior_mode,
            "course": course_name,
            "course_group": course_group,
            "noise_seed": noise_seed,
            "noise_stdev": noise_stdev,
            "goal_success": int(metrics["reached_goal"]),
            "collision_count": metrics["collisions"],
            "completion_steps": (
                world.steps if metrics["reached_goal"] else ""
            ),
            "path_length": metrics["path_distance"],
            "fitness": metrics["score"],
            "evaluation_steps": world.steps,
        })
    return records


def write_results(path, records):
    """Persist all completed episodes to a portable CSV file."""
    with Path(path).open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records, course_group=None):
    selected = [
        row for row in records
        if course_group is None or row["course_group"] == course_group
    ]
    groups = {}
    for row in selected:
        group = (row["controller"], row["behavior_mode"])
        groups.setdefault(group, []).append(row)

    scope = "training courses" if course_group == "training" else "all courses"
    print(f"\nPerformance summary ({scope}):")
    for (controller, mode), rows in sorted(groups.items()):
        successes = sum(row["goal_success"] for row in rows)
        success_rate = successes / len(rows)
        successful_steps = [
            row["completion_steps"] for row in rows if row["goal_success"]
        ]
        mean_completion = (
            sum(successful_steps) / len(successful_steps)
            if successful_steps else float("nan")
        )
        print(
            f"  {controller:14} {mode:10} "
            f"success={successes}/{len(rows)} ({success_rate:.1%})  "
            f"collisions={sum(row['collision_count'] for row in rows) / len(rows):.2f} avg  "
            f"completion={mean_completion:.1f} steps  "
            f"path={sum(row['path_length'] for row in rows) / len(rows):.2f} avg  "
            f"fitness={sum(row['fitness'] for row in rows) / len(rows):.2f} avg"
        )
        if course_group is None:
            target = "MEETS" if success_rate >= 0.8 else "below"
            print(f"    80% goal-success target: {target}")


def write_training_artifacts(genomes_path, history_path, genomes, history_rows):
    """Persist controller genomes (for reuse/training) and GA progress."""
    np.savez(genomes_path, **genomes)
    with Path(history_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history_rows)


def run_experiment(
    seeds=DEFAULT_SEEDS,
    generations=DEFAULT_GENERATIONS,
    population_size=DEFAULT_POPULATION_SIZE,
    steps=DEFAULT_STEPS,
    noise_stdev=DEFAULT_NOISE_STDEV,
    output_path="experiment_results.csv",
    genomes_path="experiment_genomes.npz",
    history_path="experiment_training_history.csv",
    network_only=True,
    robot_noise=None,
    time_limit_factor=DEFAULT_TIME_LIMIT_FACTOR,
    reference_speed=DEFAULT_REFERENCE_SPEED,
    training=DEFAULT_TRAINING,
    mazes_per_evaluation=DEFAULT_MAZES_PER_EVALUATION,
    mirror_probability=DEFAULT_MIRROR_PROBABILITY,
    light_occlusion=DEFAULT_LIGHT_OCCLUSION,
    curriculum=None,
    maze_options=None,
):
    if training not in ("random_mazes", "courses"):
        raise ValueError("training must be 'random_mazes' or 'courses'")
    robot_noise = DEFAULT_ROBOT_NOISE if robot_noise is None else robot_noise
    world_options = {"time_limit_factor": time_limit_factor, "reference_speed": reference_speed,
                   "light_occlusion": light_occlusion}
    courses = make_evaluation_courses()
    records = []
    genomes = {}
    history_rows = []

    for training_seed in range(seeds):
        algorithm = GeneticAlgorithm(
            population_size=population_size,
            hidden_size=4,
            mutation_rate=0.15,
            mutation_scale=0.2,
            seed=training_seed,
            fitness_function=(
                evaluate_course_set_outcome if network_only else evaluate_course_set
            ),
            network_only=network_only,
            light_occlusion=light_occlusion,
        )
        genomes[f"random_seed{training_seed}"] = algorithm.population[0].genome.copy()
        random_controller = algorithm.decode_genome(
            algorithm.population[0].genome
        )

        print(
            f"\nSeed {training_seed}: measuring random-controller baseline "
            "before evolution..."
        )
        baseline = evaluate_controller(
            random_controller,
            "random_init",
            "full",
            training_seed,
            courses,
            steps,
            noise_stdev,
            robot_noise,
            **world_options,
        )
        records.extend(baseline)
        write_results(output_path, records)
        print_summary(baseline, course_group="training")

        mode_label = "network only" if network_only else "with heuristics"
        training_label = (
            f"{mazes_per_evaluation} random mazes per generation"
            if training == "random_mazes" else "courses 1-4"
        )
        print(f"Seed {training_seed}: evolving on {training_label} ({mode_label})...")
        # New, reproducible mazes and noise every generation, shared by the
        # whole population so individuals are compared on the same conditions.
        training_noise_rng = np.random.default_rng(
            [TRAINING_NOISE_SEED_BASE, training_seed]
        )
        maze_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, training_seed, 1])
        generation_conditions = {}

        def conditions_for(generation):
            if generation not in generation_conditions:
                options = (curriculum_maze_options(generation, generations, curriculum)
                           if curriculum else maze_options)
                generation_conditions.clear()
                generation_conditions[generation] = (
                    sample_random_mazes(maze_rng, mazes_per_evaluation, mirror_probability,
                                        options),
                    int(training_noise_rng.integers(2**31)),
                )
            return generation_conditions[generation]

        def world_factory(controller):
            if training == "random_mazes":
                mazes, noise_seed_base = conditions_for(algorithm.generation)
                return RandomMazeWorld(
                    controller,
                    mazes=mazes,
                    noise_stdev=noise_stdev,
                    noise_seed_base=noise_seed_base,
                    network_only=network_only,
                    robot_noise=robot_noise,
                    **world_options,
                )
            noise_seed_base = int(training_noise_rng.integers(2**31))
            return MultiCourseWorld(
                controller,
                noise_stdev=noise_stdev,
                noise_seed_base=noise_seed_base,
                network_only=network_only,
                robot_noise=robot_noise,
                **world_options,
            )

        best = algorithm.run(
            generations=generations,
            steps=steps,
            world_factory=world_factory,
        )
        evolved_controller = algorithm.decode_genome(best.genome)
        genomes[f"evolved_seed{training_seed}"] = best.genome.copy()
        history_rows.extend(
            {"training_seed": training_seed, **row} for row in algorithm.history
        )
        write_training_artifacts(genomes_path, history_path, genomes, history_rows)
        print(f"  Best mean training fitness: {best.fitness:.2f}")

        for name, controller, mode in (
            ("random_init", random_controller, "neural_only"),
            ("evolved", evolved_controller, "full"),
            ("evolved", evolved_controller, "neural_only"),
            ("heuristic_only", None, "full"),
        ):
            records.extend(evaluate_controller(
                controller,
                name,
                mode,
                training_seed,
                courses,
                steps,
                noise_stdev,
                robot_noise,
                **world_options,
            ))
        write_results(output_path, records)

    print_summary(records)
    print(f"\nDetailed episode results written to {Path(output_path).resolve()}")
    print(f"Controller genomes written to {Path(genomes_path).resolve()}")
    print(f"Training history written to {Path(history_path).resolve()}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--noise-stdev", type=float, default=DEFAULT_NOISE_STDEV)
    parser.add_argument("--output", default="experiment_results.csv")
    parser.add_argument("--genomes-output", default="experiment_genomes.npz")
    parser.add_argument("--history-output", default="experiment_training_history.csv")
    parser.add_argument("--with-heuristics", action="store_true",
                        help="evolve with the hand-coded navigation heuristics enabled "
                             "(default: the network is the only controller)")
    parser.add_argument("--no-robot-noise", action="store_true",
                        help="disable sensor, motor, wheel-mismatch, and start-pose noise")
    parser.add_argument("--time-limit-factor", type=float, default=DEFAULT_TIME_LIMIT_FACTOR,
                        help="per-course step limit as a multiple of the shortest-path "
                             "driving time at --reference-speed; 0 disables (default: 8)")
    parser.add_argument("--train-on-courses", action="store_true",
                        help="train on courses 1-4 instead of fresh random mazes "
                             "(default: courses are a held-out test set only)")
    parser.add_argument("--mazes-per-evaluation", type=int,
                        default=DEFAULT_MAZES_PER_EVALUATION,
                        help="random mazes per generation, shared by the population (default: 8)")
    parser.add_argument("--curriculum", action="store_true",
                        help="grow random-maze size 12 -> 16 -> 20 over the run "
                             "(default: 12 x 12 mazes throughout)")
    parser.add_argument("--light-occlusion", choices=("blocked", "through", "none"),
                        default=DEFAULT_LIGHT_OCCLUSION,
                        help="goal-light sensors: walls block the light ('blocked', default), "
                             "walls do not ('through'), or no light inputs ('none')")
    parser.add_argument("--mirror-probability", type=float,
                        default=DEFAULT_MIRROR_PROBABILITY,
                        help="probability of mirroring each random maze left-right (default: 0)")
    parser.add_argument("--reference-speed", type=float, default=DEFAULT_REFERENCE_SPEED,
                        help="maze units per step used for per-course limits (default: 0.05)")
    arguments = parser.parse_args()
    if arguments.seeds < 1:
        parser.error("--seeds must be at least 1")
    if arguments.generations < 1:
        parser.error("--generations must be at least 1")
    if arguments.population_size < 2:
        parser.error("--population-size must be at least 2")
    if arguments.steps < 0:
        parser.error("--steps must be non-negative")
    if arguments.noise_stdev < 0:
        parser.error("--noise-stdev must be non-negative")
    if arguments.time_limit_factor < 0:
        parser.error("--time-limit-factor must be non-negative")
    if arguments.reference_speed <= 0:
        parser.error("--reference-speed must be positive")
    if arguments.mazes_per_evaluation < 1:
        parser.error("--mazes-per-evaluation must be at least 1")
    if not 0 <= arguments.mirror_probability <= 1:
        parser.error("--mirror-probability must be between 0 and 1")
    run_experiment(
        seeds=arguments.seeds,
        generations=arguments.generations,
        population_size=arguments.population_size,
        steps=arguments.steps,
        noise_stdev=arguments.noise_stdev,
        output_path=arguments.output,
        genomes_path=arguments.genomes_output,
        history_path=arguments.history_output,
        network_only=not arguments.with_heuristics,
        robot_noise={} if arguments.no_robot_noise else None,
        time_limit_factor=arguments.time_limit_factor,
        reference_speed=arguments.reference_speed,
        training="courses" if arguments.train_on_courses else "random_mazes",
        mazes_per_evaluation=arguments.mazes_per_evaluation,
        mirror_probability=arguments.mirror_probability,
        light_occlusion=None if arguments.light_occlusion == "none" else arguments.light_occlusion,
        curriculum=DEFAULT_CURRICULUM if arguments.curriculum else None,
    )


if __name__ == "__main__":
    main()
