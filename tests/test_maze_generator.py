import math
import unittest

import numpy as np

from evolution import fitness
from evolution.experiment import (
    DEFAULT_CURRICULUM,
    DEFAULT_MAZE_OPTIONS,
    DEFAULT_MAZES_PER_EVALUATION,
    DEFAULT_TIME_LIMIT_FACTOR,
    make_evaluation_courses,
    sample_random_mazes,
    RandomMazeWorld,
    course_step_limit,
    curriculum_maze_options,
    evaluate_course_set_outcome,
)
from evolution.fitness import path_distances
from sim.held_out_courses import COURSE_SPECS, make_test_course
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
            self.assertAlmostEqual(world.vehicle.x_pos, random_maze.start[0], delta=0.4 + 1e-9)
            self.assertAlmostEqual(world.vehicle.y_pos, random_maze.start[1], delta=0.4 + 1e-9)

        first.run(3000)
        self.assertEqual([world.steps for _, world in first.worlds], first.step_limits)
        result = evaluate_course_set_outcome(first)
        self.assertEqual(len(result["courses"]), 3)
        self.assertTrue(np.isfinite(result["score"]))


class CurriculumTests(unittest.TestCase):
    def test_stages_grow_maze_size_over_the_run(self):
        sizes = [curriculum_maze_options(g, 60)["size"] for g in range(60)]
        self.assertEqual(sizes[:20], [12.0] * 20)
        self.assertEqual(sizes[20:40], [16.0] * 20)
        self.assertEqual(sizes[40:], [20.0] * 20)
        self.assertEqual(curriculum_maze_options(5, 60, None), {})
        with self.assertRaises(ValueError):
            curriculum_maze_options(0, 10, ((0.5, {"size": 12.0}),))

    def test_every_stage_generates_solvable_mazes(self):
        rng = np.random.default_rng(3)
        for _, options in DEFAULT_CURRICULUM:
            for _ in range(10):
                random_maze = generate_perfect_maze(rng, **options)
                maze = random_maze.maze
                self.assertEqual(maze.width, options["size"])
                self.assertTrue(
                    options["min_route_cells"] <= random_maze.route_cells <= options["max_route_cells"]
                )
                self.assertTrue(np.isfinite(path_distances(maze, [random_maze.start], 1.0)[0]))

    def test_random_maze_world_uses_stage_options(self):
        world = RandomMazeWorld(StandStill(), np.random.default_rng(0), mazes_per_evaluation=2,
                                maze_options=curriculum_maze_options(0, 60))
        self.assertEqual([m.maze.width for m in world.mazes], [12.0, 12.0])


class HeldOutCourseTests(unittest.TestCase):
    def test_courses_are_perfect_12x12_mazes_with_4_unit_corridors(self):
        for name, spec in COURSE_SPECS.items():
            maze, start, heading = make_test_course(name)
            self.assertEqual((maze.width, maze.height), (12.0, 12.0))
            # perfect maze on 3 x 3 cells: 8 passages, 12 - 8 = 4 unit wall pieces inside
            self.assertEqual(len(spec["passages"]), 8)
            wall_units = sum(math.dist(a, b) / 4.0 for a, b in maze.interior_walls)
            self.assertAlmostEqual(wall_units, 4.0)
            self.assertGreaterEqual(maze.clearance(start, 1.0), 1.0 - 1e-9)
            self.assertTrue(np.isfinite(path_distances(maze, [start], 1.0)[0]))

    def test_course_b_mirrors_course_a(self):
        maze_a, start_a, heading_a = make_test_course("A")
        maze_b, start_b, heading_b = make_test_course("B")
        mirrored = sorted(((-x1, y1), (-x2, y2)) for (x1, y1), (x2, y2) in maze_a.interior_walls)
        self.assertEqual(sorted(tuple(sorted(w)) for w in maze_b.interior_walls),
                         sorted(tuple(sorted(w)) for w in mirrored))
        self.assertEqual(start_b, (-start_a[0], start_a[1]))
        self.assertEqual(maze_b.goal, (-maze_a.goal[0], maze_a.goal[1]))
        self.assertEqual(heading_a, heading_b)  # both face north, mirror-symmetric

    def test_evaluation_courses_are_the_held_out_set(self):
        names = [entry[0] for entry in make_evaluation_courses()]
        self.assertEqual(names, ["course_A", "course_B", "course_C"])
        self.assertTrue(all(len(entry) == 5 for entry in make_evaluation_courses()))


class TrainingDefaultsTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(DEFAULT_TIME_LIMIT_FACTOR, 8.0)
        self.assertEqual(DEFAULT_MAZES_PER_EVALUATION, 8)
        self.assertEqual(DEFAULT_MAZE_OPTIONS["size"], 12.0)
        mazes = sample_random_mazes(np.random.default_rng(0))
        self.assertEqual(len(mazes), 8)
        self.assertTrue(all(m.maze.width == 12.0 for m in mazes))

    def test_shared_mazes_give_identical_conditions(self):
        mazes = sample_random_mazes(np.random.default_rng(1), 2)
        first = RandomMazeWorld(StandStill(), mazes=mazes, noise_seed_base=5)
        second = RandomMazeWorld(StandStill(), mazes=mazes, noise_seed_base=5)
        self.assertEqual(first.step_limits, second.step_limits)
        for (_, a), (_, b) in zip(first.worlds, second.worlds):
            self.assertEqual((a.vehicle.x_pos, a.vehicle.orientation), (b.vehicle.x_pos, b.vehicle.orientation))
        with self.assertRaises(ValueError):
            RandomMazeWorld(StandStill())


if __name__ == "__main__":
    unittest.main()
