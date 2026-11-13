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


if __name__ == "__main__":
    unittest.main()
