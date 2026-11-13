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
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle
from sim.world import World

TRAINING_COURSES = (1, 2, 3, 4)
DEFAULT_SEEDS = 3
DEFAULT_GENERATIONS = 10
DEFAULT_POPULATION_SIZE = 30
DEFAULT_STEPS = 5000
DEFAULT_NOISE_STDEV = 0.05
TRAINING_NOISE_SEED_BASE = 900000
SUCCESS_BONUS = 100.0
SPEED_BONUS = 50.0
COLLISION_SCORE_FLOOR = -100.0
# Per-course time limit = factor * shortest-path length / reference speed.
# 0.05 maze units/step is ~75% of top speed (1/15); evolved network-only
# controllers averaged 0.052 on steps when they were not blocked by a wall.
# A factor of 0 (or None) disables per-course limits ("no time pressure").
DEFAULT_TIME_LIMIT_FACTOR = 2.0
DEFAULT_REFERENCE_SPEED = 0.05
# Network-only training uses freshly generated random mazes by default; the
# built-in courses are kept as a held-out test set.
DEFAULT_TRAINING = "random_mazes"
DEFAULT_MAZES_PER_EVALUATION = 3
DEFAULT_MIRROR_PROBABILITY = 0.0
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
                 reference_speed=DEFAULT_REFERENCE_SPEED):
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


class RandomMazeWorld:
    """Run one controller on freshly generated random mazes.

    Each instance samples ``mazes_per_evaluation`` new perfect mazes from
    ``maze_rng`` (see ``sim.maze_generator.generate_perfect_maze``), each with
    a random start cell and heading, and runs every maze for at most its own
    ``course_step_limit`` measured from that start. Same interface as
    ``MultiCourseWorld``, so the course-set fitness functions apply.
    """

    def __init__(self, controller, maze_rng, mazes_per_evaluation=DEFAULT_MAZES_PER_EVALUATION,
                 mirror_probability=DEFAULT_MIRROR_PROBABILITY, noise_stdev=0.0,
                 noise_seed_base=TRAINING_NOISE_SEED_BASE, network_only=True, robot_noise=None,
                 time_limit_factor=DEFAULT_TIME_LIMIT_FACTOR,
                 reference_speed=DEFAULT_REFERENCE_SPEED, maze_options=None):
        if mazes_per_evaluation < 1:
            raise ValueError("mazes_per_evaluation must be at least 1")
        robot_noise = DEFAULT_ROBOT_NOISE if robot_noise is None else robot_noise
        maze_options = {} if maze_options is None else maze_options
        self.mazes = [
            generate_perfect_maze(maze_rng, mirror_probability=mirror_probability, **maze_options)
            for _ in range(mazes_per_evaluation)
        ]
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
    """Return four training courses and two distinct, held-out layouts."""
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
):
    """Evaluate one controller on every course with reproducible noise.

    ``robot_noise`` (default ``DEFAULT_ROBOT_NOISE``) and the per-course step
    limits apply to every controller and mode so comparisons share the same
    conditions.
    """
    robot_noise = DEFAULT_ROBOT_NOISE if robot_noise is None else robot_noise
    use_heuristics = behavior_mode == "full"
    records = []
    for course_index, (course_name, course_group, maze) in enumerate(courses):
        noise_seed = training_seed * 10000 + course_index
        vehicle = Vehicle(
            controller=controller,
            noise_stdev=noise_stdev,
            noise_seed=noise_seed,
            navigation_heuristics=use_heuristics,
            **robot_noise,
        )
        world = World(maze=maze, vehicle=vehicle)
        world.run(_episode_steps(
            steps, course_step_limit(maze, time_limit_factor, reference_speed)
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
):
    if training not in ("random_mazes", "courses"):
        raise ValueError("training must be 'random_mazes' or 'courses'")
    robot_noise = DEFAULT_ROBOT_NOISE if robot_noise is None else robot_noise
    time_limits = {"time_limit_factor": time_limit_factor, "reference_speed": reference_speed}
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
            **time_limits,
        )
        records.extend(baseline)
        write_results(output_path, records)
        print_summary(baseline, course_group="training")

        mode_label = "network only" if network_only else "with heuristics"
        training_label = (
            f"{mazes_per_evaluation} fresh random mazes per evaluation"
            if training == "random_mazes" else "courses 1-4"
        )
        print(f"Seed {training_seed}: evolving on {training_label} ({mode_label})...")
        # Fresh, reproducible noise (and mazes) for every training evaluation,
        # so start poses, sensor/motor noise, and layouts cannot be memorized.
        training_noise_rng = np.random.default_rng(
            [TRAINING_NOISE_SEED_BASE, training_seed]
        )
        maze_rng = np.random.default_rng([TRAINING_NOISE_SEED_BASE, training_seed, 1])

        def world_factory(controller):
            noise_seed_base = int(training_noise_rng.integers(2**31))
            if training == "random_mazes":
                return RandomMazeWorld(
                    controller,
                    maze_rng,
                    mazes_per_evaluation=mazes_per_evaluation,
                    mirror_probability=mirror_probability,
                    noise_stdev=noise_stdev,
                    noise_seed_base=noise_seed_base,
                    network_only=network_only,
                    robot_noise=robot_noise,
                    **time_limits,
                )
            return MultiCourseWorld(
                controller,
                noise_stdev=noise_stdev,
                noise_seed_base=noise_seed_base,
                network_only=network_only,
                robot_noise=robot_noise,
                **time_limits,
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
                **time_limits,
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
                             "driving time at --reference-speed; 0 disables (default: 2)")
    parser.add_argument("--train-on-courses", action="store_true",
                        help="train on courses 1-4 instead of fresh random mazes "
                             "(default: courses are a held-out test set only)")
    parser.add_argument("--mazes-per-evaluation", type=int,
                        default=DEFAULT_MAZES_PER_EVALUATION,
                        help="random mazes sampled for each training evaluation (default: 3)")
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
    )


if __name__ == "__main__":
    main()
