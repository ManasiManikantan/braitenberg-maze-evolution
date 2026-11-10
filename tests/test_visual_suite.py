"""Broad test suite for the maze-evolution project, with visual outputs.

Run everything (saves figures to ``test_output/``):

    python tests/test_visual_suite.py

Open each figure in a window as well:

    python tests/test_visual_suite.py --show

Run only the figure-producing tests or only the plain ones:

    python -m unittest tests.test_visual_suite.VisualTests
    python -m unittest tests.test_visual_suite.MazeTests

Watch the live animations (each window blocks until you close it):

    $env:VISUAL_SHOW = "1"; python -m unittest tests.test_visual_suite.AnimationTests

Without --show / VISUAL_SHOW=1 the animation tests render frames headlessly.
"""

import math
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SHOW = "--show" in sys.argv or os.environ.get("VISUAL_SHOW") == "1"
if "--show" in sys.argv:
    sys.argv.remove("--show")

import matplotlib

if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from controller.network import NeuralNetwork
from evolution.experiment import (
    evaluate_controller,
    make_evaluation_courses,
)
from evolution.fitness import evaluate_fitness
from evolution.genetic_algorithm import GeneticAlgorithm
from evolution.map_elites import MapElites, plot_archive
from sim.maze import Maze
from sim.robot import Vehicle
from sim.visualization import animate_world
from sim.world import World

OUTPUT_DIR = PROJECT_ROOT / "test_output"
SENSOR_RANGE_SLACK = 1e-6


def empty_maze(goal=(8.0, 0.0)):
    return Maze(walls=[], obstacles=[], goal=goal, goal_radius=1.0)


def run_world(maze, steps, controller=None, noise=0.0, seed=0, heuristics=True):
    vehicle = Vehicle(
        controller=controller,
        noise_stdev=noise,
        noise_seed=seed,
        navigation_heuristics=heuristics,
    )
    world = World(maze=maze, vehicle=vehicle)
    world.run(steps)
    return world


def draw_maze(axis, maze, title=None):
    for (x0, y0), (x1, y1) in maze.walls:
        axis.plot((x0, x1), (y0, y1), color="#25313c", linewidth=2.5,
                  solid_capstyle="round")
    for x0, y0, x1, y1 in maze.obstacles:
        axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 facecolor="#d6dce1", edgecolor="#596570"))
    axis.add_patch(Circle(maze.goal, maze.goal_radius, facecolor="#78b88a",
                          edgecolor="#247245", alpha=0.45))
    axis.plot(*maze.goal, marker="*", color="#247245", markersize=10)
    axis.set_xlim(maze.x_min - 0.5, maze.x_max + 0.5)
    axis.set_ylim(maze.y_min - 0.5, maze.y_max + 0.5)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    if title:
        axis.set_title(title, fontsize=9)


def finish_figure(figure, name):
    """Save a figure under test_output/ and optionally display it."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    figure.savefig(path, dpi=130, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(figure)
    return path


class FixedController:
    def __init__(self, left, right):
        self.outputs = (left, right)

    def forward(self, sensors):
        return self.outputs


class MazeTests(unittest.TestCase):
    def test_all_four_courses_build_and_differ(self):
        layouts = {
            (tuple(m.interior_walls), tuple(m.obstacles))
            for m in (Maze(difficulty=d) for d in (1, 2, 3, 4))
        }
        self.assertEqual(len(layouts), 4)

    def test_goal_is_never_inside_a_wall_on_any_course(self):
        for _, _, maze in make_evaluation_courses():
            self.assertFalse(maze.collides(maze.goal, maze.goal_radius))

    def test_start_position_is_free_on_every_course(self):
        for name, _, maze in make_evaluation_courses():
            with self.subTest(course=name):
                self.assertFalse(maze.collides((0.0, 0.0), 1.0))

    def test_boundary_walls_block_the_edge(self):
        maze = empty_maze()
        self.assertTrue(maze.collides((maze.x_max, 0.0), 0.5))
        self.assertFalse(maze.collides((0.0, 0.0), 0.5))

    def test_raycast_hits_boundary_at_expected_distance(self):
        maze = empty_maze()
        distance = maze.raycast((0.0, 0.0), (1.0, 0.0), 100.0)
        self.assertAlmostEqual(distance, maze.x_max, places=3)

    def test_raycast_is_capped_by_max_distance(self):
        maze = empty_maze()
        self.assertAlmostEqual(maze.raycast((0.0, 0.0), (1.0, 0.0), 3.0), 3.0)

    def test_raycast_hits_interior_wall(self):
        maze = Maze(walls=[((4.0, -5.0), (4.0, 5.0))], obstacles=[],
                    goal=(8.0, 8.0), goal_radius=1.0)
        self.assertAlmostEqual(maze.raycast((0.0, 0.0), (1.0, 0.0), 50.0), 4.0, places=3)

    def test_raycast_hits_obstacle(self):
        maze = Maze(walls=[], obstacles=[(3.0, -1.0, 4.0, 1.0)],
                    goal=(8.0, 8.0), goal_radius=1.0)
        self.assertAlmostEqual(maze.raycast((0.0, 0.0), (1.0, 0.0), 50.0), 3.0, places=3)

    def test_clearance_is_smaller_near_a_wall(self):
        maze = empty_maze()
        self.assertLess(maze.clearance((maze.x_max - 1.0, 0.0), 0.5),
                        maze.clearance((0.0, 0.0), 0.5))

    def test_collides_with_obstacle_but_not_beside_it(self):
        maze = Maze(walls=[], obstacles=[(3.0, -1.0, 4.0, 1.0)],
                    goal=(8.0, 8.0), goal_radius=1.0)
        self.assertTrue(maze.collides((3.5, 0.0), 0.5))
        self.assertFalse(maze.collides((3.5, 5.0), 0.5))

    def test_overlapping_goal_and_wall_is_rejected(self):
        with self.assertRaises(ValueError):
            Maze(walls=[((7.0, -5.0), (7.0, 5.0))], obstacles=[],
                 goal=(7.0, 0.0), goal_radius=1.0)

    def test_nonpositive_goal_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            Maze(goal_radius=0.0)


class NetworkTests(unittest.TestCase):
    def test_forward_returns_two_finite_motor_values(self):
        output = NeuralNetwork(seed=1).forward([0.8, 0.5, 0.2])
        self.assertEqual(len(output), 2)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_same_seed_gives_identical_outputs(self):
        sensors = [0.3, 0.6, 0.9]
        np.testing.assert_array_equal(
            NeuralNetwork(seed=5).forward(sensors),
            NeuralNetwork(seed=5).forward(sensors),
        )

    def test_different_seeds_give_different_outputs(self):
        sensors = [0.3, 0.6, 0.9]
        self.assertFalse(np.allclose(
            NeuralNetwork(seed=1).forward(sensors),
            NeuralNetwork(seed=2).forward(sensors),
        ))

    def test_weight_shapes_follow_hidden_size(self):
        network = NeuralNetwork(hidden_size=7, seed=0)
        self.assertEqual(network.weights_input_hidden.shape, (3, 7))
        self.assertEqual(network.weights_hidden_output.shape, (7, 2))

    def test_outputs_respond_to_sensor_changes(self):
        network = NeuralNetwork(seed=3)
        self.assertFalse(np.allclose(
            network.forward([0.0, 0.0, 0.0]), network.forward([1.0, 1.0, 1.0])
        ))


class VehicleTests(unittest.TestCase):
    def test_vehicle_starts_at_origin_facing_positive_x(self):
        vehicle = Vehicle(noise_stdev=0)
        self.assertEqual((vehicle.x_pos, vehicle.y_pos), (0.0, 0.0))
        self.assertEqual(vehicle.orientation, 0.0)

    def test_equal_motor_outputs_drive_straight_forward(self):
        vehicle = Vehicle(noise_stdev=0)
        vehicle.left_motor = vehicle.right_motor = 0.6
        vehicle.move()
        self.assertGreater(vehicle.x_pos, 0.0)
        self.assertAlmostEqual(vehicle.y_pos, 0.0)
        self.assertAlmostEqual(vehicle.orientation, 0.0)

    def test_unequal_motor_outputs_turn_the_vehicle(self):
        vehicle = Vehicle(noise_stdev=0)
        vehicle.left_motor, vehicle.right_motor = 0.2, 0.8
        vehicle.move()
        self.assertNotAlmostEqual(vehicle.orientation, 0.0)

    def test_noise_seed_gives_reproducible_motion(self):
        poses = []
        for _ in range(2):
            vehicle = Vehicle(noise_stdev=0.1, noise_seed=9)
            vehicle.left_motor = vehicle.right_motor = 0.5
            for _ in range(20):
                vehicle.move()
            poses.append((vehicle.x_pos, vehicle.y_pos, vehicle.orientation))
        self.assertEqual(poses[0], poses[1])

    def test_different_noise_seeds_diverge(self):
        poses = []
        for seed in (1, 2):
            vehicle = Vehicle(noise_stdev=0.1, noise_seed=seed)
            vehicle.left_motor = vehicle.right_motor = 0.5
            for _ in range(20):
                vehicle.move()
            poses.append(vehicle.orientation)
        self.assertNotEqual(poses[0], poses[1])

    def test_zero_noise_is_deterministic_regardless_of_seed(self):
        orientations = set()
        for seed in (1, 2, 3):
            vehicle = Vehicle(noise_stdev=0, noise_seed=seed)
            vehicle.left_motor, vehicle.right_motor = 0.3, 0.7
            vehicle.move()
            orientations.add(vehicle.orientation)
        self.assertEqual(len(orientations), 1)

    def test_sensors_report_distances_within_range(self):
        maze = empty_maze()
        vehicle = Vehicle(noise_stdev=0)
        vehicle.sense(maze, maze.goal)
        for distance in (vehicle.left_distance, vehicle.front_distance,
                         vehicle.right_distance):
            self.assertGreater(distance, 0.0)
            self.assertLessEqual(distance, vehicle.sensor_range + SENSOR_RANGE_SLACK)

    def test_front_sensor_sees_a_wall_ahead(self):
        maze = Maze(walls=[((3.0, -5.0), (3.0, 5.0))], obstacles=[],
                    goal=(8.0, 8.0), goal_radius=1.0)
        vehicle = Vehicle(noise_stdev=0)
        vehicle.sense(maze, maze.goal)
        self.assertLess(vehicle.front_distance, vehicle.sensor_range)
        self.assertAlmostEqual(vehicle.front_distance, 3.0 - vehicle.radius, delta=0.2)

    def test_neural_only_mode_uses_controller_outputs_directly(self):
        maze = empty_maze()
        vehicle = Vehicle(controller=FixedController(0.8, 0.4), noise_stdev=0,
                          navigation_heuristics=False)
        vehicle.sense(maze, maze.goal)
        vehicle.think()
        self.assertEqual((vehicle.left_motor, vehicle.right_motor), (0.8, 0.4))

    def test_neural_only_mode_clips_motor_outputs(self):
        maze = empty_maze()
        vehicle = Vehicle(controller=FixedController(5.0, -5.0), noise_stdev=0,
                          navigation_heuristics=False)
        vehicle.sense(maze, maze.goal)
        vehicle.think()
        self.assertLessEqual(abs(vehicle.left_motor), 1.0)
        self.assertLessEqual(abs(vehicle.right_motor), 1.0)

    def test_heuristics_avoid_a_wall_that_neural_only_drives_into(self):
        maze = Maze(walls=[((3.0, -8.0), (3.0, 8.0))], obstacles=[],
                    goal=(8.0, 8.0), goal_radius=1.0)
        forward = FixedController(0.9, 0.9)
        guided = run_world(maze, 300, forward, heuristics=True)
        blind = run_world(maze, 300, forward, heuristics=False)
        self.assertLess(guided.collisions, blind.collisions)

    def test_avoidance_scan_request_is_ignored_without_heuristics(self):
        vehicle = Vehicle(noise_stdev=0, navigation_heuristics=False)
        vehicle.request_avoidance_scan()
        self.assertFalse(vehicle._turning_to_clear)


class WorldTests(unittest.TestCase):
    def test_zero_steps_leaves_world_untouched(self):
        world = run_world(empty_maze(), 0)
        self.assertEqual(world.steps, 0)
        self.assertEqual(len(world.position_history), 1)

    def test_history_grows_with_each_step(self):
        world = run_world(empty_maze(goal=(-8.0, 8.0)), 25,
                          FixedController(0.5, 0.5))
        self.assertEqual(world.steps, 25)
        self.assertEqual(len(world.position_history), 26)

    def test_straight_robot_reaches_a_goal_ahead(self):
        maze = empty_maze(goal=(5.0, 0.0))
        world = run_world(maze, 3000, FixedController(0.8, 0.8))
        self.assertTrue(world.reached_goal)

    def test_robot_does_not_reach_a_goal_behind_it_when_driving_straight(self):
        maze = empty_maze(goal=(-8.0, 0.0))
        world = run_world(maze, 200, FixedController(0.8, 0.8),
                          heuristics=False)
        self.assertFalse(world.reached_goal)

    def test_collisions_are_counted(self):
        maze = Maze(walls=[((3.0, -8.0), (3.0, 8.0))], obstacles=[],
                    goal=(8.0, 8.0), goal_radius=1.0)
        world = run_world(maze, 300, FixedController(0.9, 0.9), heuristics=False)
        self.assertGreater(world.collisions, 0)

    def test_world_run_is_reproducible_with_noise_seed(self):
        maze = Maze(difficulty=1)
        first = run_world(maze, 400, noise=0.05, seed=4)
        second = run_world(Maze(difficulty=1), 400, noise=0.05, seed=4)
        self.assertEqual(first.position_history, second.position_history)

    def test_position_stays_inside_the_maze(self):
        maze = Maze(difficulty=2)
        world = run_world(maze, 800, noise=0.05, seed=1)
        for x, y in world.position_history:
            self.assertTrue(maze.x_min <= x <= maze.x_max)
            self.assertTrue(maze.y_min <= y <= maze.y_max)


class FitnessTests(unittest.TestCase):
    def test_metrics_include_every_reported_quantity(self):
        metrics = evaluate_fitness(run_world(Maze(difficulty=1), 200))
        for key in ("score", "collisions", "path_distance", "reached_goal",
                    "behavior_descriptor"):
            self.assertIn(key, metrics)

    def test_descriptor_values_are_normalized(self):
        coverage, progress = evaluate_fitness(
            run_world(Maze(difficulty=1), 500))["behavior_descriptor"]
        self.assertTrue(0.0 <= coverage <= 1.0)
        self.assertTrue(0.0 <= progress <= 1.0)

    def test_reaching_the_goal_outscores_not_moving(self):
        maze = empty_maze(goal=(5.0, 0.0))
        success = evaluate_fitness(run_world(maze, 3000, FixedController(0.8, 0.8)))
        idle = evaluate_fitness(run_world(empty_maze(goal=(5.0, 0.0)), 3000,
                                          FixedController(0.0, 0.0),
                                          heuristics=False))
        self.assertTrue(success["reached_goal"])
        self.assertGreater(success["score"], idle["score"])

    def test_collisions_lower_the_score(self):
        wall = Maze(walls=[((3.0, -8.0), (3.0, 8.0))], obstacles=[],
                    goal=(8.0, 8.0), goal_radius=1.0)
        crashing = evaluate_fitness(
            run_world(wall, 300, FixedController(0.9, 0.9), heuristics=False))
        careful = evaluate_fitness(
            run_world(Maze(walls=[((3.0, -8.0), (3.0, 8.0))], obstacles=[],
                           goal=(8.0, 8.0), goal_radius=1.0),
                      300, FixedController(0.9, 0.9), heuristics=True))
        self.assertGreater(careful["score"], crashing["score"])

    def test_path_length_is_nonnegative_and_grows_with_steps(self):
        short = evaluate_fitness(run_world(empty_maze(goal=(-8.0, 8.0)), 20,
                                           FixedController(0.5, 0.5)))
        long = evaluate_fitness(run_world(empty_maze(goal=(-8.0, 8.0)), 80,
                                          FixedController(0.5, 0.5)))
        self.assertGreaterEqual(short["path_distance"], 0.0)
        self.assertGreater(long["path_distance"], short["path_distance"])

    def test_invalid_parameters_are_rejected(self):
        world = run_world(empty_maze(), 5)
        with self.assertRaises(ValueError):
            evaluate_fitness(world, stuck_window=0)
        with self.assertRaises(ValueError):
            evaluate_fitness(world, cell_size=0)


class GeneticAlgorithmTests(unittest.TestCase):
    def test_genome_size_matches_network_topology(self):
        algorithm = GeneticAlgorithm(population_size=4, hidden_size=3, seed=1)
        self.assertEqual(algorithm.genome_size, 6 * 3 + 2)
        self.assertEqual(len(algorithm.population), 4)

    def test_decode_then_encode_round_trips(self):
        algorithm = GeneticAlgorithm(population_size=4, hidden_size=3, seed=1)
        genome = algorithm.population[0].genome
        network = algorithm.decode_genome(genome)
        np.testing.assert_allclose(algorithm.encode_network(network), genome)

    def test_mutation_preserves_genome_size_and_changes_values(self):
        algorithm = GeneticAlgorithm(population_size=4, mutation_rate=1.0,
                                     mutation_scale=0.5, seed=2)
        genome = algorithm.population[0].genome
        mutated = algorithm.mutate(genome)
        self.assertEqual(mutated.shape, genome.shape)
        self.assertFalse(np.allclose(mutated, genome))

    def test_zero_mutation_rate_leaves_genome_unchanged(self):
        algorithm = GeneticAlgorithm(population_size=4, mutation_rate=0.0, seed=2)
        genome = algorithm.population[0].genome
        np.testing.assert_array_equal(algorithm.mutate(genome), genome)

    def test_crossover_children_mix_both_parents(self):
        algorithm = GeneticAlgorithm(population_size=4, crossover_rate=1.0, seed=3)
        size = algorithm.genome_size
        first, second = np.zeros(size), np.ones(size)
        child_one, child_two = algorithm.crossover(first, second)
        self.assertTrue(0 < child_one.sum() < size)
        self.assertAlmostEqual(child_one.sum() + child_two.sum(), size)

    def test_wrong_size_genome_is_rejected(self):
        algorithm = GeneticAlgorithm(population_size=4, seed=1)
        with self.assertRaises(ValueError):
            algorithm.mutate(np.zeros(3))

    def test_invalid_constructor_arguments_are_rejected(self):
        for kwargs in ({"population_size": 1}, {"hidden_size": 0},
                       {"mutation_rate": 1.5}, {"tournament_size": 99}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    GeneticAlgorithm(**{"population_size": 4, **kwargs})

    def test_population_stays_the_same_size_across_generations(self):
        algorithm = GeneticAlgorithm(population_size=6, seed=1,
                                     fitness_function=lambda world: 1.0)
        algorithm.run(generations=3, steps=0)
        self.assertEqual(len(algorithm.population), 6)

    def test_best_fitness_never_decreases(self):
        algorithm = GeneticAlgorithm(population_size=6, seed=1)
        algorithm.run(generations=3, steps=60)
        best = [row["best_fitness"] for row in algorithm.history]
        self.assertEqual(best, sorted(best))

    def test_run_is_reproducible_for_the_same_seed(self):
        scores = []
        for _ in range(2):
            algorithm = GeneticAlgorithm(population_size=6, seed=7)
            scores.append(algorithm.run(generations=2, steps=60).fitness)
        self.assertEqual(scores[0], scores[1])

    def test_history_has_one_row_per_generation(self):
        algorithm = GeneticAlgorithm(population_size=4, seed=1,
                                     fitness_function=lambda world: 1.0)
        algorithm.run(generations=5, steps=0)
        self.assertEqual(len(algorithm.history), 5)

    def test_zero_generations_is_rejected(self):
        with self.assertRaises(ValueError):
            GeneticAlgorithm(population_size=4).run(generations=0)


class ExperimentTests(unittest.TestCase):
    def test_evaluation_set_has_four_training_and_two_held_out_courses(self):
        groups = [group for _, group, _ in make_evaluation_courses()]
        self.assertEqual(groups, ["training"] * 4 + ["held_out"] * 2)

    def test_records_contain_all_requested_metrics(self):
        records = evaluate_controller(
            NeuralNetwork(seed=1), "random_init", "full", 0,
            make_evaluation_courses()[:2], steps=100, noise_stdev=0.05)
        self.assertEqual(len(records), 2)
        for record in records:
            for key in ("goal_success", "collision_count", "completion_steps",
                        "path_length", "fitness"):
                self.assertIn(key, record)

    def test_completion_steps_blank_when_goal_not_reached(self):
        record = evaluate_controller(
            NeuralNetwork(seed=1), "random_init", "full", 0,
            make_evaluation_courses()[:1], steps=0, noise_stdev=0.0)[0]
        self.assertEqual(record["goal_success"], 0)
        self.assertEqual(record["completion_steps"], "")

    def test_evaluation_is_reproducible(self):
        runs = [
            evaluate_controller(None, "heuristic_only", "full", 3,
                                make_evaluation_courses()[:2], 300, 0.05)
            for _ in range(2)
        ]
        self.assertEqual(runs[0], runs[1])

    def test_different_training_seeds_use_different_noise(self):
        courses = make_evaluation_courses()[:1]
        seeds = {
            evaluate_controller(None, "h", "full", seed, courses, 0, 0.05)[0]["noise_seed"]
            for seed in (0, 1, 2)
        }
        self.assertEqual(len(seeds), 3)


class MapElitesTests(unittest.TestCase):
    def test_cell_mapping_clips_to_grid_edges(self):
        archive = MapElites(bins=(10, 5), coverage_range=(0.0, 0.2), progress_max=1.0)
        self.assertEqual(archive.cell_for((0.0, 0.0)), (0, 0))
        self.assertEqual(archive.cell_for((0.1, 0.5)), (5, 2))
        self.assertEqual(archive.cell_for((9.0, 9.0)), (9, 4))
        self.assertEqual(archive.cell_for((-1.0, -1.0)), (0, 0))

    def test_better_genome_replaces_worse_in_same_cell(self):
        archive = MapElites(seed=1)
        scores = iter((1.0, 5.0, 3.0))
        archive.evaluate = lambda genome: {
            "genome": genome, "fitness": next(scores), "descriptor": (0.1, 0.5)}
        outcomes = [archive.add(np.zeros(3)) for _ in range(3)]
        self.assertEqual(outcomes, [True, True, False])
        self.assertEqual(next(iter(archive.archive.values()))["fitness"], 5.0)

    def test_different_descriptors_fill_different_cells(self):
        archive = MapElites(seed=1)
        descriptors = iter(((0.05, 0.1), (0.15, 0.7)))
        archive.evaluate = lambda genome: {
            "genome": genome, "fitness": 1.0, "descriptor": next(descriptors)}
        archive.add(np.zeros(3))
        archive.add(np.zeros(3))
        self.assertEqual(len(archive.archive), 2)

    def test_run_respects_evaluation_budget(self):
        archive = MapElites(seed=2, noise_stdev=0.0)
        archive.run(evaluations=10, initial=5, steps=100, log_every=1000)
        self.assertEqual(archive.evaluations, 10)
        self.assertGreaterEqual(len(archive.archive), 1)

    def test_stats_report_fill_fraction(self):
        archive = MapElites(seed=2, noise_stdev=0.0)
        archive.run(evaluations=8, initial=4, steps=100, log_every=1000)
        stats = archive.stats()
        self.assertEqual(stats["cells"], 100)
        self.assertAlmostEqual(stats["coverage"], stats["filled"] / 100)

    def test_search_is_reproducible(self):
        runs = []
        for _ in range(2):
            archive = MapElites(seed=3)
            archive.run(evaluations=8, initial=4, steps=100, log_every=1000)
            runs.append(sorted((c, e["fitness"]) for c, e in archive.archive.items()))
        self.assertEqual(runs[0], runs[1])

    def test_invalid_bins_are_rejected(self):
        with self.assertRaises(ValueError):
            MapElites(bins=(0, 5))


class VisualTests(unittest.TestCase):
    """Each test renders a figure to test_output/ and checks it was written."""

    def assert_saved(self, path):
        self.assertTrue(path.exists(), path)
        self.assertGreater(path.stat().st_size, 1000)

    def test_visual_course_gallery(self):
        courses = make_evaluation_courses()
        figure, axes = plt.subplots(2, 3, figsize=(12, 8))
        for axis, (name, group, maze) in zip(axes.flat, courses):
            draw_maze(axis, maze, f"{name} ({group})")
            axis.plot(0, 0, "o", color="#4d91c6", markersize=8)
        figure.suptitle("Training courses and held-out layouts (blue dot = start)")
        self.assert_saved(finish_figure(figure, "01_course_gallery.png"))

    def test_visual_heuristic_trajectories(self):
        courses = make_evaluation_courses()
        figure, axes = plt.subplots(2, 3, figsize=(12, 8))
        for axis, (name, _, maze) in zip(axes.flat, courses):
            world = run_world(maze, 3000, noise=0.05, seed=1)
            path = np.array(world.position_history)
            draw_maze(axis, maze)
            axis.plot(path[:, 0], path[:, 1], color="#d58a32", linewidth=1.3)
            axis.plot(*path[-1], "o", color="#174c70")
            outcome = f"reached goal in {world.steps}" if world.reached_goal else "no goal"
            axis.set_title(f"{name}: {outcome}, {world.collisions} collisions",
                           fontsize=8)
        figure.suptitle("Hand-coded heuristics only, noise 0.05")
        self.assert_saved(finish_figure(figure, "02_heuristic_trajectories.png"))

    def test_visual_sensor_rays(self):
        maze = Maze(difficulty=2)
        figure, axis = plt.subplots(figsize=(7, 7))
        draw_maze(axis, maze, "Sensor rays at three robot poses")
        for x, y, heading in ((0.0, 0.0, 0.0), (-6.0, -6.0, math.pi / 2),
                              (4.0, 4.0, math.pi)):
            vehicle = Vehicle(noise_stdev=0)
            vehicle.x_pos, vehicle.y_pos, vehicle.orientation = x, y, heading
            vehicle.update_sensor_pos()
            vehicle.sense(maze, maze.goal)
            axis.add_patch(Circle((x, y), vehicle.radius, facecolor="#4d91c6",
                                  edgecolor="#174c70", alpha=0.8, zorder=5))
            for sensor, distance, color in (
                (vehicle.ls, vehicle.left_distance, "#d58a32"),
                (vehicle.fs, vehicle.front_distance, "#bd5b45"),
                (vehicle.rs, vehicle.right_distance, "#347e9e"),
            ):
                axis.plot((sensor.x_pos, sensor.x_pos + distance * math.cos(heading)),
                          (sensor.y_pos, sensor.y_pos + distance * math.sin(heading)),
                          color=color, linewidth=2)
                self.assertLessEqual(distance, vehicle.sensor_range + SENSOR_RANGE_SLACK)
        self.assert_saved(finish_figure(figure, "03_sensor_rays.png"))

    def test_visual_neural_network_response(self):
        network = NeuralNetwork(seed=1)
        levels = np.linspace(0.0, 1.0, 40)
        left, right = [], []
        for level in levels:
            output = network.forward([level, 0.5, 1.0 - level])
            left.append(output[0])
            right.append(output[1])
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(levels, left, label="left motor")
        axis.plot(levels, right, label="right motor")
        axis.set_xlabel("left sensor level (right = 1 - left, front = 0.5)")
        axis.set_ylabel("motor output")
        axis.set_title("Random network's sensor-to-motor response")
        axis.legend()
        axis.grid(alpha=0.3)
        self.assertEqual(len(left), 40)
        self.assert_saved(finish_figure(figure, "04_network_response.png"))

    def test_visual_ga_learning_curve(self):
        algorithm = GeneticAlgorithm(population_size=10, hidden_size=4, seed=0,
                                     mutation_rate=0.15, mutation_scale=0.2)
        algorithm.run(generations=6, steps=400)
        generations = [row["generation"] for row in algorithm.history]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(generations, [r["best_fitness"] for r in algorithm.history],
                  marker="o", label="best so far")
        axis.plot(generations, [r["generation_best_fitness"] for r in algorithm.history],
                  marker="s", label="generation best")
        axis.plot(generations, [r["mean_fitness"] for r in algorithm.history],
                  marker="^", label="population mean")
        axis.set_xlabel("generation")
        axis.set_ylabel("fitness")
        axis.set_title("Genetic algorithm progress (small demo run)")
        axis.legend()
        axis.grid(alpha=0.3)
        self.assertEqual(len(generations), 6)
        self.assert_saved(finish_figure(figure, "05_ga_learning_curve.png"))

    def test_visual_controller_comparison(self):
        courses = make_evaluation_courses()
        algorithm = GeneticAlgorithm(population_size=8, hidden_size=4, seed=0,
                                     mutation_rate=0.15, mutation_scale=0.2)
        algorithm.run(generations=3, steps=500)
        evolved = algorithm.decode_genome(algorithm.best_individual.genome)
        random_net = algorithm.decode_genome(algorithm.population[0].genome)
        variants = {
            "heuristics only": (None, "full"),
            "random net + heuristics": (random_net, "full"),
            "evolved net + heuristics": (evolved, "full"),
            "evolved net only": (evolved, "neural_only"),
        }
        results = {}
        for label, (controller, mode) in variants.items():
            records = evaluate_controller(controller, label, mode, 0, courses,
                                          1500, 0.05)
            results[label] = {
                "Goal success rate": np.mean([r["goal_success"] for r in records]),
                "Collisions (capped at 50)": np.mean(
                    [min(r["collision_count"], 50) for r in records]),
                "Path length": np.mean([r["path_length"] for r in records]),
                "Fitness (clipped at -100)": np.mean(
                    [max(r["fitness"], -100) for r in records]),
            }
        metrics = list(next(iter(results.values())))
        figure, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
        colors = ["#4d91c6", "#9aa5b1", "#78b88a", "#d58a32"]
        for axis, metric in zip(axes, metrics):
            axis.bar(range(len(results)), [results[v][metric] for v in results],
                     color=colors)
            axis.set_xticks(range(len(results)))
            axis.set_xticklabels(list(results), rotation=30, ha="right", fontsize=7)
            axis.set_title(metric, fontsize=9)
        figure.suptitle("Ablation: what the network adds beyond the heuristics "
                        "(short demo budget)")
        self.assert_saved(finish_figure(figure, "06_controller_comparison.png"))

    def test_visual_noise_robustness(self):
        maze_factory = lambda: Maze(difficulty=1)
        noise_levels = (0.0, 0.05, 0.1, 0.2)
        seeds = range(4)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        success_rates = []
        for noise in noise_levels:
            outcomes = [run_world(maze_factory(), 3000, noise=noise, seed=s)
                        for s in seeds]
            success_rates.append(np.mean([w.reached_goal for w in outcomes]))
            if noise in (0.0, 0.2):
                for world in outcomes:
                    path = np.array(world.position_history)
                    axes[1].plot(path[:, 0], path[:, 1], alpha=0.6,
                                 color="#4d91c6" if noise == 0.0 else "#bd5b45")
        axes[0].bar([str(n) for n in noise_levels], success_rates, color="#78b88a")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_xlabel("movement noise stdev")
        axes[0].set_ylabel("goal success rate")
        axes[0].set_title("Success vs noise (course 1, 4 seeds)")
        draw_maze(axes[1], maze_factory(), "Paths: blue = no noise, red = noise 0.2")
        self.assertTrue(all(0.0 <= rate <= 1.0 for rate in success_rates))
        self.assert_saved(finish_figure(figure, "07_noise_robustness.png"))

    def test_visual_seed_variability(self):
        maze_factory = lambda: Maze(difficulty=4)
        figure, axis = plt.subplots(figsize=(7, 7))
        draw_maze(axis, maze_factory(), "Same controller, 6 noise seeds (course 4)")
        for seed in range(6):
            world = run_world(maze_factory(), 3000, noise=0.05, seed=seed)
            path = np.array(world.position_history)
            axis.plot(path[:, 0], path[:, 1], linewidth=1.2,
                      label=f"seed {seed}{' (goal)' if world.reached_goal else ''}")
        axis.legend(fontsize=7, loc="lower left")
        self.assert_saved(finish_figure(figure, "08_seed_variability.png"))

    def test_visual_map_elites_archive(self):
        archive = MapElites(difficulty=1, seed=0, noise_stdev=0.05)
        archive.run(evaluations=60, initial=30, steps=1500, log_every=1000)
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "09_map_elites_archive.png"
        plot_archive(archive, path, show=SHOW)
        self.assertGreaterEqual(len(archive.archive), 1)
        self.assert_saved(path)


class AnimationTests(unittest.TestCase):
    """Live playback with --show; headless frame-rendering checks otherwise."""

    HEADLESS_FRAMES = (0, 5, 25)

    def render_frames(self, animation):
        """Draw a few frames without opening a window and return the final one."""
        animation._init_draw()
        for frame in self.HEADLESS_FRAMES:
            animation._draw_frame(frame)
        return animation

    def play(self, world, steps, title):
        animation = animate_world(world, steps=steps, interval=15, show=False)
        animation._fig.axes[0].set_title(title)
        if SHOW:
            plt.show()
        else:
            self.render_frames(animation)
            plt.close(animation._fig)
        return animation

    def test_animate_each_course_with_heuristics(self):
        for difficulty in (1, 2, 3, 4):
            with self.subTest(course=difficulty):
                world = World(
                    maze=Maze(difficulty=difficulty),
                    vehicle=Vehicle(noise_stdev=0.05, noise_seed=1),
                )
                self.play(world, 3000, f"Course {difficulty}: heuristics only")
                if not SHOW:
                    self.assertGreaterEqual(world.steps, self.HEADLESS_FRAMES[-1])

    def test_animate_held_out_courses(self):
        for name, _, maze in make_evaluation_courses()[4:]:
            with self.subTest(course=name):
                world = World(maze=maze,
                              vehicle=Vehicle(noise_stdev=0.05, noise_seed=2))
                self.play(world, 3000, f"Held-out course: {name}")
                if not SHOW:
                    self.assertGreater(len(world.position_history), 1)

    def test_animate_evolved_controller(self):
        algorithm = GeneticAlgorithm(population_size=8, hidden_size=4, seed=0,
                                     mutation_rate=0.15, mutation_scale=0.2)
        algorithm.run(generations=3, steps=500)
        controller = algorithm.decode_genome(algorithm.best_individual.genome)
        world = World(maze=Maze(difficulty=1),
                      vehicle=Vehicle(controller=controller, noise_stdev=0.05,
                                      noise_seed=1))
        self.play(world, 3000, "Course 1: briefly evolved network + heuristics")
        if not SHOW:
            self.assertGreater(len(world.position_history), 1)

    def test_animate_neural_only_robot(self):
        network = NeuralNetwork(seed=3)
        world = World(maze=Maze(difficulty=1),
                      vehicle=Vehicle(controller=network, noise_stdev=0.05,
                                      noise_seed=1, navigation_heuristics=False))
        self.play(world, 1500, "Course 1: random network only, no heuristics")
        if not SHOW:
            self.assertGreater(len(world.position_history), 1)

    def test_animate_side_by_side_ablation(self):
        """Two windows: identical robot and network, heuristics on versus off."""
        network = NeuralNetwork(seed=3)
        animations, worlds = [], []
        for heuristics, label in ((True, "WITH heuristics"),
                                  (False, "network ONLY")):
            world = World(maze=Maze(difficulty=1),
                          vehicle=Vehicle(controller=network, noise_stdev=0.05,
                                          noise_seed=1,
                                          navigation_heuristics=heuristics))
            animation = animate_world(world, steps=3000, interval=15, show=False)
            animation._fig.axes[0].set_title(f"Ablation: {label}")
            animations.append(animation)
            worlds.append(world)
        if SHOW:
            plt.show()
        else:
            for animation in animations:
                self.render_frames(animation)
                plt.close(animation._fig)
            self.assertEqual(worlds[0].steps, worlds[1].steps)

    def test_animation_stops_advancing_once_goal_is_reached(self):
        maze = empty_maze(goal=(3.0, 0.0))
        world = World(maze=maze, vehicle=Vehicle(
            controller=FixedController(0.8, 0.8), noise_stdev=0))
        animation = animate_world(world, steps=2000, interval=15, show=False)
        animation._init_draw()
        animation._draw_frame(1500)
        steps_at_goal = world.steps
        animation._draw_frame(1800)
        plt.close(animation._fig)
        self.assertTrue(world.reached_goal)
        self.assertEqual(world.steps, steps_at_goal)

    def test_animation_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            animate_world(World(), steps=-1, show=False)
        with self.assertRaises(ValueError):
            animate_world(World(), interval=0, show=False)


if __name__ == "__main__":
    unittest.main(verbosity=1)
