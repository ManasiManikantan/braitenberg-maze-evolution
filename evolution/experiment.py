"""Reproducible baseline, evolution, transfer, and heuristic-ablation study."""

import argparse
import csv
from pathlib import Path

import numpy as np

from evolution.fitness import evaluate_fitness
from evolution.genetic_algorithm import GeneticAlgorithm
from sim.maze import Maze
from sim.robot import Vehicle
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


class MultiCourseWorld:
    """Run one controller on each training course and expose mean fitness."""

    def __init__(self, controller, noise_stdev=0.0, noise_seed_base=TRAINING_NOISE_SEED_BASE):
        self.worlds = [
            (
                f"course_{difficulty}",
                World(
                    maze=Maze(difficulty=difficulty),
                    vehicle=Vehicle(
                        controller=controller,
                        noise_stdev=noise_stdev,
                        noise_seed=noise_seed_base + difficulty,
                    ),
                ),
            )
            for difficulty in TRAINING_COURSES
        ]

    def run(self, steps):
        self.steps = steps
        for _, world in self.worlds:
            world.run(steps)


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
):
    """Evaluate one controller on every course with reproducible noise."""
    use_heuristics = behavior_mode == "full"
    records = []
    for course_index, (course_name, course_group, maze) in enumerate(courses):
        noise_seed = training_seed * 10000 + course_index
        vehicle = Vehicle(
            controller=controller,
            noise_stdev=noise_stdev,
            noise_seed=noise_seed,
            navigation_heuristics=use_heuristics,
        )
        world = World(maze=maze, vehicle=vehicle)
        world.run(steps)
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
):
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
            fitness_function=evaluate_course_set,
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
        )
        records.extend(baseline)
        write_results(output_path, records)
        print_summary(baseline, course_group="training")

        print(f"Seed {training_seed}: evolving on courses 1-4...")
        best = algorithm.run(
            generations=generations,
            steps=steps,
            world_factory=lambda controller: MultiCourseWorld(
                controller, noise_stdev=noise_stdev
            ),
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
    run_experiment(
        seeds=arguments.seeds,
        generations=arguments.generations,
        population_size=arguments.population_size,
        steps=arguments.steps,
        noise_stdev=arguments.noise_stdev,
        output_path=arguments.output,
        genomes_path=arguments.genomes_output,
        history_path=arguments.history_output,
    )


if __name__ == "__main__":
    main()
