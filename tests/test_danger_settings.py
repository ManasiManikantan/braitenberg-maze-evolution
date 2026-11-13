import unittest

import numpy as np

from sim.maze import Maze
from sim.robot import Vehicle
from sim.world import World


class Fixed:
    def __init__(self, left, right):
        self.output = np.array([left, right])

    def forward(self, inputs):
        return self.output


def box(goal=(-7.0, 7.0)):
    return Maze(width=20.0, height=20.0, walls=[], obstacles=[], goal=goal, goal_radius=1.0)


def vehicle(controller, **options):
    return Vehicle(controller=controller, noise_stdev=0.0, navigation_heuristics=False, **options)


class WallsHurtTests(unittest.TestCase):
    def test_pinned_against_a_wall_is_one_hit(self):
        world = World(maze=box(), vehicle=vehicle(Fixed(1.0, 1.0)))
        world.run(400)                         # drives into the +x wall and stays pinned
        self.assertGreater(world.collisions, 50)
        self.assertEqual(world.contact_entries, 1)
        self.assertFalse(world.dead)

    def test_health_runs_out_and_ends_the_run(self):
        world = World(maze=box(), vehicle=vehicle(Fixed(1.0, 1.0)), health=1)
        world.run(400)
        self.assertTrue(world.dead)
        self.assertFalse(world.reached_goal)
        steps = world.steps
        world.run(50)
        self.assertEqual(world.steps, steps)   # dead robots do not move

    def test_contact_steps_drain_health(self):
        world = World(maze=box(), vehicle=vehicle(Fixed(1.0, 1.0)), health=10, contact_step_cost=0.5)
        world.run(400)                         # one hit (1 + 0.5) then 0.5 per pinned step
        self.assertTrue(world.dead)
        self.assertEqual(world.contact_entries, 1)
        self.assertEqual(world.collisions, 18)  # 1.5 + 17 * 0.5 = 10

    def test_health_off_by_default(self):
        world = World(maze=box(), vehicle=vehicle(Fixed(1.0, 1.0)))
        self.assertIsNone(world.health)


class HiddenLightTests(unittest.TestCase):
    def test_light_only_visible_within_range(self):
        maze = box(goal=(6.0, 0.0))
        near = vehicle(Fixed(0, 0), light_occlusion="blocked", light_visible_distance=8.0)
        far = vehicle(Fixed(0, 0), light_occlusion="blocked", light_visible_distance=3.0)
        normal = vehicle(Fixed(0, 0), light_occlusion="blocked")
        for v in (near, far, normal):
            v.sense(maze, maze.goal)
        self.assertGreater(near.left_light + near.right_light, 0.0)
        self.assertEqual(far.left_light + far.right_light, 0.0)
        self.assertAlmostEqual(near.left_light, normal.left_light)


class BatteryTests(unittest.TestCase):
    def test_energy_runs_out_at_motor_cost(self):
        world = World(maze=box(), vehicle=vehicle(Fixed(0.5, 0.5)), energy=10.0)
        world.run(100)
        self.assertTrue(world.out_of_energy and world.stopped)
        self.assertEqual(world.steps, 10)                 # 1.0 energy per step
        self.assertAlmostEqual(world.energy_used, 10.0)

    def test_energy_per_step_metric(self):
        from evolution.metrics import record_run, run_metrics
        result = run_metrics(record_run(Fixed(0.25, -0.5), box(), max_steps=40, light_occlusion=None,
                                        noise_stdev=0.0, robot_noise={}))
        self.assertAlmostEqual(result["energy_per_step"], 0.75)


class DeadEndGoalTests(unittest.TestCase):
    def test_goal_moves_to_a_dead_end(self):
        from sim.maze_generator import dead_end_cells, generate_perfect_maze, with_goal_in_dead_end
        for seed in range(5):
            rng = np.random.default_rng(seed)
            plain = generate_perfect_maze(rng, size=12.0, min_route_cells=2, max_route_cells=8,
                                          mirror_probability=0.5)
            moved = with_goal_in_dead_end(plain, rng)
            ends = dead_end_cells(plain.maze)
            self.assertGreaterEqual(len(ends), 2)
            self.assertIn(tuple(moved.maze.goal), [tuple(e) for e in ends])
            self.assertNotEqual(tuple(moved.maze.goal), tuple(plain.start))
            self.assertEqual(len(moved.maze.walls), len(plain.maze.walls))


if __name__ == "__main__":
    unittest.main()
