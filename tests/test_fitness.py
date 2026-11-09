import math
import unittest

from evolution.fitness import evaluate_fitness
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


if __name__ == "__main__":
    unittest.main()