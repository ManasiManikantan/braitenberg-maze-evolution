import unittest

import numpy as np

from controller.network import NeuralNetwork
from controller.wall_follower import WallFollower
from evolution.metrics import (
    BEHAVIOR_MEASURES,
    measure_controller,
    random_maze_episodes,
    record_run,
    run_metrics,
)
from sim.maze import Maze
from sim.maze_generator import generate_perfect_maze

NO_NOISE = {"noise_stdev": 0.0, "robot_noise": {}}


class Fixed:
    def __init__(self, left, right):
        self.output = np.array([left, right])

    def forward(self, inputs):
        return self.output


class Mirrored:
    """Left-right mirror of a controller: swap left/right inputs and outputs."""

    def __init__(self, base):
        self.base = base

    def forward(self, inputs):
        left, front, right = inputs[:3]
        mirrored = [right, front, left]
        if len(inputs) == 5:
            mirrored += [inputs[4], inputs[3]]
        left_motor, right_motor = self.base.forward(np.array(mirrored))
        return np.array([right_motor, left_motor])


def open_arena():
    return Maze(width=40.0, height=40.0, walls=[], obstacles=[], goal=(18.0, 18.0), goal_radius=1.0)


class TurningBiasTests(unittest.TestCase):
    def test_constant_turners_have_opposite_full_bias(self):
        left = run_metrics(record_run(Fixed(0.4, 0.6), open_arena(), max_steps=300,
                                      light_occlusion=None, **NO_NOISE))
        right = run_metrics(record_run(Fixed(0.6, 0.4), open_arena(), max_steps=300,
                                       light_occlusion=None, **NO_NOISE))
        self.assertAlmostEqual(left["turning_bias"], 1.0)    # faster right wheel turns left
        self.assertAlmostEqual(right["turning_bias"], -1.0)

    def test_mirrored_controller_in_mirrored_maze_has_opposite_bias(self):
        for seed in range(3):
            plain = generate_perfect_maze(np.random.default_rng(seed), size=12.0,
                                          min_route_cells=2, max_route_cells=8)
            mirrored = generate_perfect_maze(np.random.default_rng(seed), size=12.0,
                                             min_route_cells=2, max_route_cells=8,
                                             mirror_probability=1.0)
            network = NeuralNetwork(seed=seed)
            a = run_metrics(record_run(network, plain.maze, plain.start, plain.start_heading,
                                       max_steps=800, light_occlusion=None, **NO_NOISE))
            b = run_metrics(record_run(Mirrored(network), mirrored.maze, mirrored.start,
                                       mirrored.start_heading, max_steps=800, light_occlusion=None,
                                       **NO_NOISE))
            self.assertGreater(abs(a["turning_bias"]), 0.05)
            self.assertAlmostEqual(a["turning_bias"], -b["turning_bias"], places=6)
            for name in ("speed", "wall_clearance", "exploration", "pct_steps_in_collision"):
                self.assertAlmostEqual(a[name], b[name], places=6)
            # The 0.25-unit shortest-path grid is not exactly mirror-symmetric
            # (a goal centre on a grid line falls in the cell to its right).
            self.assertAlmostEqual(a["path_progress"], b["path_progress"], delta=0.03)

    def test_left_and_right_wall_followers_have_clearly_opposite_bias(self):
        # A left-hand follower keeps walls on its left, so it circulates
        # clockwise around the free space: net RIGHT turning (negative bias),
        # and the right-hand follower the reverse.
        episodes = random_maze_episodes(12, seed=5, noise_seed_base=100)
        left = measure_controller(WallFollower("left"), episodes)["mean"]["turning_bias"]
        right = measure_controller(WallFollower("right"), episodes)["mean"]["turning_bias"]
        self.assertLess(left, -0.05)
        self.assertGreater(right, 0.05)


class MeasureTests(unittest.TestCase):
    def test_standing_still(self):
        random_maze = generate_perfect_maze(np.random.default_rng(2), size=12.0,
                                            min_route_cells=2, max_route_cells=8)
        record = record_run(Fixed(0.0, 0.0), random_maze.maze, random_maze.start,
                            random_maze.start_heading, max_steps=50, light_occlusion=None,
                            **NO_NOISE)
        result = run_metrics(record)
        self.assertEqual(result["speed"], 0.0)
        self.assertEqual(result["turning_bias"], 0.0)
        self.assertAlmostEqual(result["exploration"], 1 / 9)   # one of nine 4-unit cells
        self.assertEqual(result["path_progress"], 0.0)
        self.assertEqual(result["steps"], 50.0)
        self.assertEqual(result["goal_reached"], 0.0)

    def test_full_speed_straight_line(self):
        result = run_metrics(record_run(Fixed(1.0, 1.0), open_arena(), max_steps=100,
                                        light_occlusion=None, **NO_NOISE))
        self.assertAlmostEqual(result["speed"], 1.0)
        self.assertEqual(result["pct_steps_in_collision"], 0.0)
        self.assertGreater(result["wall_clearance"], 0.0)

    def test_measures_are_in_range_and_summarized(self):
        episodes = random_maze_episodes(4, seed=1)
        summary = measure_controller(NeuralNetwork(seed=3, input_size=5), episodes)
        self.assertEqual(len(summary["runs"]), 4)
        for name in BEHAVIOR_MEASURES:
            self.assertIn(name, summary["mean"])
            self.assertGreaterEqual(summary["std"][name], 0.0)
        for run in summary["runs"]:
            self.assertTrue(0.0 <= run["speed"] <= 1.0)
            self.assertTrue(0.0 <= run["wall_clearance"] <= 1.0)
            self.assertTrue(-1.0 <= run["turning_bias"] <= 1.0)
            self.assertTrue(0.0 < run["exploration"] <= 1.0)
            self.assertTrue(0.0 <= run["pct_steps_in_collision"] <= 100.0)


if __name__ == "__main__":
    unittest.main()
