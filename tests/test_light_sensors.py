import math
import unittest

import numpy as np

from evolution.experiment import RandomMazeWorld
from evolution.genetic_algorithm import GeneticAlgorithm
from sim.maze import Maze
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle
from sim.world import World


def open_maze(goal, walls=()):
    return Maze(walls=list(walls), obstacles=[], goal=goal, goal_radius=1.0)


def lights(goal, walls=(), heading=0.0, **kwargs):
    vehicle = Vehicle(noise_stdev=0, start_heading=heading, **{"light_occlusion": "through", **kwargs})
    vehicle.sense(open_maze(goal, walls), goal)
    return vehicle.left_light, vehicle.right_light


class LightSensorTests(unittest.TestCase):
    def test_light_is_off_by_default(self):
        vehicle = Vehicle(noise_stdev=0)
        vehicle.sense(open_maze((5.0, 0.0)), (5.0, 0.0))
        self.assertEqual(len(vehicle.controller_inputs()), 3)
        self.assertEqual((vehicle.left_light, vehicle.right_light), (0.0, 0.0))

    def test_goal_ahead_lights_both_sensors_equally(self):
        left, right = lights((6.0, 0.0))
        self.assertAlmostEqual(left, right)
        self.assertGreater(left, 0.0)
        self.assertLessEqual(left, 1.0)

    def test_goal_on_one_side_lights_that_sensor_more(self):
        left, right = lights((3.0, 4.0))
        self.assertGreater(left, right)
        left, right = lights((3.0, -4.0))
        self.assertGreater(right, left)

    def test_goal_behind_reads_zero(self):
        self.assertEqual(lights((-6.0, 0.0)), (0.0, 0.0))

    def test_intensity_falls_off_with_distance(self):
        near, _ = lights((3.0, 0.0))
        far, _ = lights((8.0, 0.0))
        self.assertGreater(near, far)
        # left sensor at 45 deg on the body edge; check the formula directly
        sensor = (math.cos(math.pi / 4), math.sin(math.pi / 4))
        dx, dy = 8.0 - sensor[0], 0.0 - sensor[1]
        d = math.hypot(dx, dy)
        cosine = (dx * math.cos(math.pi / 4) + dy * math.sin(math.pi / 4)) / d
        self.assertAlmostEqual(far, cosine / (1 + (d / 5.0)**2))

    def test_occlusion_modes(self):
        wall = [((4.0, -5.0), (4.0, 5.0))]
        blocked = lights((7.0, 0.0), wall, light_occlusion="blocked")
        through = lights((7.0, 0.0), wall, light_occlusion="through")
        self.assertEqual(blocked, (0.0, 0.0))
        self.assertGreater(min(through), 0.0)
        self.assertEqual(lights((7.0, 0.0), light_occlusion="blocked"),
                         lights((7.0, 0.0), light_occlusion="through"))

    def test_light_noise_is_seeded_and_clipped(self):
        readings = set()
        for seed in range(10):
            vehicle = Vehicle(noise_stdev=0, light_occlusion="through", light_noise_stdev=0.02,
                              noise_seed=seed)
            vehicle.sense(open_maze((6.0, 0.0)), (6.0, 0.0))
            self.assertTrue(0.0 <= vehicle.left_light <= 1.0)
            readings.add(vehicle.left_light)
        self.assertEqual(len(readings), 10)
        self.assertEqual(DEFAULT_ROBOT_NOISE["light_noise_stdev"], 0.02)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            Vehicle(light_occlusion="sometimes")
        with self.assertRaises(ValueError):
            GeneticAlgorithm(population_size=4, light_occlusion="sometimes")


class FiveInputNetworkTests(unittest.TestCase):
    def test_light_adds_two_network_inputs(self):
        plain = GeneticAlgorithm(population_size=4, hidden_size=4, seed=1)
        lit = GeneticAlgorithm(population_size=4, hidden_size=4, seed=1, light_occlusion="blocked")
        self.assertEqual(plain.genome_size, 26)
        self.assertEqual(lit.genome_size, 34)
        network = lit.decode_genome(lit.population[0].genome)
        self.assertEqual(network.weights_input_hidden.shape, (5, 4))
        np.testing.assert_allclose(lit.encode_network(network), lit.population[0].genome)

        world = lit._default_world_factory(network)
        self.assertEqual(world.vehicle.light_occlusion, "blocked")
        world.run(20)
        self.assertEqual(len(world.vehicle.controller_inputs()), 5)

    def test_random_maze_world_passes_light_mode(self):
        lit = GeneticAlgorithm(population_size=4, seed=2, light_occlusion="through")
        network = lit.decode_genome(lit.population[0].genome)
        mazes = RandomMazeWorld(network, np.random.default_rng(0), mazes_per_evaluation=2,
                                light_occlusion="through")
        mazes.run(50)
        for _, world in mazes.worlds:
            self.assertEqual(world.vehicle.light_occlusion, "through")


if __name__ == "__main__":
    unittest.main()
