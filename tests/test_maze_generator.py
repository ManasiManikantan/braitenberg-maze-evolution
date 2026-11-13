import math
import unittest

import numpy as np

from evolution import fitness
from evolution.experiment import RandomMazeWorld, course_step_limit, evaluate_course_set_outcome
from evolution.fitness import path_distances
from sim.maze_generator import generate_perfect_maze
from sim.robot import Vehicle


class StandStill:
    def forward(self, sensors):
        return (0.0, 0.0)


class MazeGeneratorTests(unittest.TestCase):
    def test_mazes_are_perfect_and_solvable_with_wide_corridors(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            random_maze = generate_perfect_maze(rng)
            maze = random_maze.maze
            self.assertEqual((maze.width, maze.height), (20.0, 20.0))
            self.assertEqual(maze.obstacles, [])
            self.assertTrue(3 <= random_maze.route_cells <= 6)
            self.assertFalse(maze.collides(random_maze.start, 1.0))
            self.assertTrue(np.isfinite(path_distances(maze, [random_maze.start], 1.0)[0]))
            self.assertTrue(0.0 <= random_maze.start_heading < 2 * math.pi)
            # corridors are 4 units: a robot (diameter 2) has 1 unit spare on each side
            self.assertGreaterEqual(maze.clearance(random_maze.start, 1.0), 1.0 - 1e-9)

    def test_spanning_tree_has_no_loops(self):
        # A perfect maze on a 5 x 5 grid has 25 - 1 = 24 passages; every other
        # internal edge (40 in total) is a wall, i.e. 16 unit-length wall pieces.
        rng = np.random.default_rng(5)
        for _ in range(10):
            maze = generate_perfect_maze(rng).maze
            wall_units = sum(
                math.dist(start, end) / 4.0 for start, end in maze.interior_walls
            )
            self.assertAlmostEqual(wall_units, 16.0)

    def test_generation_is_reproducible_and_varied(self):
        first = generate_perfect_maze(np.random.default_rng(7))
        second = generate_perfect_maze(np.random.default_rng(7))
        self.assertEqual(first.maze.interior_walls, second.maze.interior_walls)
        self.assertEqual((first.start, first.start_heading), (second.start, second.start_heading))
        rng = np.random.default_rng(8)
        layouts = {tuple(generate_perfect_maze(rng).maze.interior_walls) for _ in range(10)}
        self.assertEqual(len(layouts), 10)

    def test_mirroring_flips_layout_left_right(self):
        plain = generate_perfect_maze(np.random.default_rng(11), mirror_probability=0.0)
        mirrored = generate_perfect_maze(np.random.default_rng(11), mirror_probability=1.0)
        self.assertFalse(plain.mirrored)
        self.assertTrue(mirrored.mirrored)
        flipped = sorted(((-x1, y1), (-x2, y2)) for (x1, y1), (x2, y2) in plain.maze.interior_walls)
        self.assertEqual(sorted(mirrored.maze.interior_walls), flipped)
        self.assertEqual(mirrored.maze.goal, (-plain.maze.goal[0], plain.maze.goal[1]))
        self.assertEqual(mirrored.start, (-plain.start[0], plain.start[1]))
        self.assertAlmostEqual(math.cos(mirrored.start_heading), -math.cos(plain.start_heading))
        self.assertAlmostEqual(math.sin(mirrored.start_heading), math.sin(plain.start_heading))

    def test_vehicle_start_pose_with_jitter_on_top(self):
        vehicle = Vehicle(start_position=(3.0, -2.0), start_heading=1.0)
        self.assertEqual((vehicle.x_pos, vehicle.y_pos, vehicle.orientation), (3.0, -2.0, 1.0))
        jittered = Vehicle(start_position=(3.0, -2.0), start_heading=1.0, noise_seed=2,
                           start_position_stdev=0.2, start_heading_stdev=0.1)
        self.assertLessEqual(abs(jittered.x_pos - 3.0), 0.4)
        self.assertLessEqual(abs(jittered.orientation - 1.0), 0.2)
        self.assertNotEqual(jittered.x_pos, 3.0)

    def test_path_map_cache_is_bounded(self):
        rng = np.random.default_rng(1)
        for _ in range(fitness.PATH_MAP_CACHE_SIZE + 20):
            random_maze = generate_perfect_maze(rng)
            path_distances(random_maze.maze, [random_maze.start], 1.0)
        self.assertLessEqual(len(fitness._PATH_MAP_CACHE), fitness.PATH_MAP_CACHE_SIZE)


class RandomMazeWorldTests(unittest.TestCase):
    def test_each_evaluation_samples_fresh_mazes_with_their_own_limits(self):
        rng = np.random.default_rng(4)
        first = RandomMazeWorld(StandStill(), rng, mazes_per_evaluation=3)
        second = RandomMazeWorld(StandStill(), rng, mazes_per_evaluation=3)
        self.assertEqual(len(first.worlds), 3)
        self.assertNotEqual(
            [m.maze.interior_walls for m in first.mazes],
            [m.maze.interior_walls for m in second.mazes],
        )
        for random_maze, (_, world), limit in zip(first.mazes, first.worlds, first.step_limits):
            self.assertEqual(limit, course_step_limit(random_maze.maze, start=random_maze.start))
            self.assertAlmostEqual(world.vehicle.x_pos, random_maze.start[0], delta=0.4)
            self.assertAlmostEqual(world.vehicle.y_pos, random_maze.start[1], delta=0.4)

        first.run(3000)
        self.assertEqual([world.steps for _, world in first.worlds], first.step_limits)
        result = evaluate_course_set_outcome(first)
        self.assertEqual(len(result["courses"]), 3)
        self.assertTrue(np.isfinite(result["score"]))


if __name__ == "__main__":
    unittest.main()
