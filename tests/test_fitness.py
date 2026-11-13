import math
import unittest

import numpy as np

from evolution import fitness as fitness_module
from evolution.fitness import evaluate_fitness, evaluate_outcome_fitness, path_distances
from sim.maze import Maze
from sim.robot import Vehicle
from sim.world import World


class FitnessTests(unittest.TestCase):
    def make_world(self, positions):
        maze = Maze(
            walls=[],
            obstacles=[(2.0, -1.0, 3.0, 1.0)],
            goal=(8.0, -7.0),
            goal_radius=1.0,
        )
        world = World(maze=maze, vehicle=Vehicle(noise_stdev=0))
        world.position_history = positions
        world.steps = len(positions) - 1
        world.moving_steps = world.steps
        world.initial_distance = math.hypot(positions[0][0] - 8, positions[0][1] + 7)
        world.reached_goal = False
        return world

    def test_clearance_accounts_for_robot_radius(self):
        maze = Maze(
            walls=[],
            obstacles=[(2.0, -1.0, 3.0, 1.0)],
            goal=(8.0, -7.0),
            goal_radius=1.0,
        )

        self.assertAlmostEqual(maze.clearance((0.0, 0.0), radius=0.5), 1.5)

    def test_safe_distance_score_rewards_more_clearance(self):
        near_wall = evaluate_fitness(self.make_world([(0.0, 0.0)]))
        farther_from_wall = evaluate_fitness(self.make_world([(-5.0, 0.0)]))

        self.assertGreater(
            farther_from_wall["safe_distance_score"],
            near_wall["safe_distance_score"],
        )

    def test_repeated_collisions_have_escalating_cost(self):
        world = self.make_world([(0.0, 0.0), (0.1, 0.0)])
        world.collisions = 1
        one_collision = evaluate_fitness(world)
        world.collisions = 3
        repeated_collisions = evaluate_fitness(world)

        self.assertEqual(one_collision["repeated_collision_cost"], 0.0)
        self.assertEqual(repeated_collisions["repeated_collision_cost"], 4.0)
        self.assertLess(repeated_collisions["score"], one_collision["score"])

    def test_goal_progress_beats_standing_still(self):
        standing_still = evaluate_fitness(self.make_world([(0.0, 0.0)]))
        moving_toward_goal = evaluate_fitness(
            self.make_world([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        )

        self.assertGreater(moving_toward_goal["score"], standing_still["score"])
        self.assertGreater(moving_toward_goal["path_distance"], 0.0)
        self.assertTrue(standing_still["stuck"])

    def test_back_and_forth_motion_loses_to_goal_progress(self):
        oscillating = evaluate_fitness(
            self.make_world([
                (0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)
            ]),
            stuck_window=1,
        )
        progressing = evaluate_fitness(
            self.make_world([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]),
            stuck_window=1,
        )

        self.assertLess(oscillating["score"], progressing["score"])
        self.assertGreater(oscillating["goal_regression_distance"], 0.0)

    def test_repeated_path_is_penalized_against_new_area(self):
        loop = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]
        looping_metrics = evaluate_fitness(self.make_world(loop * 4))
        exploring_metrics = evaluate_fitness(
            self.make_world([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)])
        )

        self.assertGreater(looping_metrics["path_revisits"], 0)
        self.assertEqual(exploring_metrics["path_revisits"], 0)
        self.assertLess(looping_metrics["score"], exploring_metrics["score"])

    def test_reaching_goal_receives_large_bonus(self):
        approaching_goal = self.make_world([(0.0, 0.0)])
        reached_goal = self.make_world([(0.0, 0.0), (8.0, -5.1)])

        approaching_metrics = evaluate_fitness(approaching_goal)
        reached_metrics = evaluate_fitness(reached_goal)
        self.assertGreater(reached_metrics["score"], approaching_metrics["score"])
        self.assertEqual(reached_metrics["goal_bonus_awarded"], 5.0)
        self.assertLessEqual(reached_metrics["goal_center_error"], 2.0)
        self.assertAlmostEqual(reached_metrics["goal_overlap_margin"], 0.1)
        self.assertAlmostEqual(
            reached_metrics["score"], sum(reached_metrics["score_components"].values())
        )

    def test_movement_time_cost_is_small_and_reported(self):
        world = self.make_world([(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)])

        metrics = evaluate_fitness(world)

        self.assertAlmostEqual(metrics["score_components"]["movement_time"], -0.002)

    def test_goal_radius_edge_does_not_get_completion_bonus(self):
        world = self.make_world([(0.0, 0.0), (8.0, -4.9)])

        metrics = evaluate_fitness(world)

        self.assertFalse(metrics["reached_goal"])
        self.assertEqual(metrics["goal_bonus_awarded"], 0.0)
        self.assertLess(metrics["goal_overlap_margin"], 0.0)


class OutcomeFitnessTests(unittest.TestCase):
    def make_world(self, positions, walls=(), goal=(8.0, -7.0), collisions=0, reached=False):
        maze = Maze(walls=list(walls), obstacles=[], goal=goal, goal_radius=1.0)
        world = World(maze=maze, vehicle=Vehicle(noise_stdev=0))
        world.position_history = list(positions)
        world.steps = len(positions) - 1
        world.moving_steps = world.steps
        world.collisions = collisions
        world.reached_goal = reached
        return world

    def test_path_distance_follows_walls_not_straight_line(self):
        maze = Maze(walls=[((0.0, -10.0), (0.0, 5.0))], obstacles=[], goal=(5.0, 0.0),
                    goal_radius=1.0)
        start_distance = path_distances(maze, [(-5.0, 0.0)], radius=1.0)[0]
        # must go around the top of the wall (y > 5 + radius), not straight through
        self.assertGreater(start_distance, 20.0)

    def test_gap_exactly_robot_diameter_stays_connected_and_finite(self):
        # Wall ends at y=8; the boundary is at y=10, so the gap is exactly 2 units,
        # the robot's diameter. Course 1 has the same kind of pocket.
        maze = Maze(walls=[((0.0, -10.0), (0.0, 8.0))], obstacles=[], goal=(5.0, 0.0),
                    goal_radius=1.0)
        start, squeezing, pressed = path_distances(
            maze, [(-5.0, 0.0), (0.0, 9.0), (-1.05, 0.0)], radius=1.0)
        self.assertTrue(np.isfinite([start, squeezing, pressed]).all())
        self.assertGreater(start, 20.0)
        self.assertLess(squeezing, start)

    def test_every_course_position_has_finite_path_distance(self):
        for difficulty in (1, 2, 3, 4):
            maze = Maze(difficulty=difficulty)
            grid = [(x, y) for x in np.linspace(-9.9, 9.9, 60) for y in np.linspace(-9.9, 9.9, 60)]
            self.assertTrue(np.isfinite(path_distances(maze, grid, radius=1.0)).all())

    def test_progress_is_fraction_of_start_path_distance(self):
        world = self.make_world([(0.0, 0.0), (4.0, -3.5)])
        metrics = evaluate_outcome_fitness(world)
        start = metrics["start_path_distance"]
        expected = (start - metrics["final_path_distance"]) / start
        self.assertAlmostEqual(metrics["path_progress"], expected)
        self.assertGreater(metrics["path_progress"], 0.3)
        self.assertLess(metrics["path_progress"], 0.7)

    def test_reaching_goal_scores_full_progress_plus_bonus(self):
        world = self.make_world([(0.0, 0.0), (7.5, -6.5)], reached=True)
        metrics = evaluate_outcome_fitness(world, goal_bonus=1.0)
        self.assertEqual(metrics["path_progress"], 1.0)
        self.assertAlmostEqual(metrics["score"], 2.0)
        self.assertFalse(metrics["stuck"])

    def test_collisions_and_driving_style_are_not_scored(self):
        straight = [(0.0, 0.0), (2.0, -2.0), (4.0, -3.5)]
        wiggly = [(0.0, 0.0), (-3.0, 3.0), (5.0, 2.0), (-2.0, -6.0), (4.0, -3.5)]
        calm = evaluate_outcome_fitness(self.make_world(straight, collisions=0))
        crashing = evaluate_outcome_fitness(self.make_world(straight, collisions=50))
        meandering = evaluate_outcome_fitness(self.make_world(wiggly, collisions=0))
        self.assertAlmostEqual(calm["score"], crashing["score"])
        self.assertAlmostEqual(calm["score"], meandering["score"])
        self.assertEqual(crashing["collisions"], 50)
        self.assertGreater(meandering["unique_cells_visited"], calm["unique_cells_visited"])
        self.assertEqual(set(calm["score_components"]),
                         {"path_progress", "goal_completion", "stuck"})

    def test_stuck_penalty_after_configurable_steps_without_path_progress(self):
        positions = [(0.0, 0.0), (2.0, -2.0)] + [(2.0, -2.0)] * 50
        progress_only = evaluate_outcome_fitness(self.make_world(positions), stuck_steps=60)
        stuck = evaluate_outcome_fitness(self.make_world(positions), stuck_steps=50,
                                         stuck_penalty=0.2)
        self.assertFalse(progress_only["stuck"])
        self.assertTrue(stuck["stuck"])
        self.assertEqual(stuck["steps_without_progress"], 50)
        self.assertAlmostEqual(progress_only["score"] - stuck["score"], 0.2)

    def test_old_name_still_refers_to_shaped_fitness(self):
        self.assertIs(fitness_module.evaluate_fitness, fitness_module.evaluate_shaped_fitness)


if __name__ == "__main__":
    unittest.main()