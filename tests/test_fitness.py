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

    def test_wall_following_prefers_parallel_safe_motion(self):
        maze = Maze(
            walls=[((0.0, -5.0), (0.0, 5.0))],
            obstacles=[],
            goal=(8.0, -7.0),
            goal_radius=1.0,
        )
        alongside = maze.wall_following_score(
            (1.0, -1.0), (1.0, 1.0), radius=0.25
        )
        toward_wall = maze.wall_following_score(
            (1.0, -1.0), (3.0, -1.0), radius=0.25
        )

        self.assertGreater(alongside, 0.0)
        self.assertEqual(toward_wall, 0.0)

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

    def test_reaching_goal_receives_large_bonus(self):
        approaching_goal = self.make_world([(0.0, 0.0)])
        reached_goal = self.make_world([(0.0, 0.0), (8.0, -7.0)])

        self.assertGreater(evaluate_fitness(reached_goal)["score"],
                           evaluate_fitness(approaching_goal)["score"])


if __name__ == "__main__":
    unittest.main()