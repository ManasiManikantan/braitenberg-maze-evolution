import unittest

import numpy as np

from evolution.experiment import MultiCourseWorld
from evolution.genetic_algorithm import GeneticAlgorithm
from sim.maze import Maze
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle
from sim.world import World


class FixedController:
    def __init__(self, left, right):
        self.output = (left, right)

    def forward(self, sensors):
        return self.output


def run(vehicle, steps=50, maze=None):
    world = World(maze=Maze(walls=[], obstacles=[]) if maze is None else maze, vehicle=vehicle)
    world.run(steps)
    return world


class RobotNoiseTests(unittest.TestCase):
    def test_noise_is_off_by_default_and_draws_nothing(self):
        plain = Vehicle(noise_seed=4, controller=FixedController(0.6, 0.4),
                        navigation_heuristics=False)
        explicit_zero = Vehicle(noise_seed=4, controller=FixedController(0.6, 0.4),
                                navigation_heuristics=False, sensor_noise_stdev=0.0,
                                motor_noise_stdev=0.0, motor_mismatch=0.0,
                                start_position_stdev=0.0, start_heading_stdev=0.0)
        self.assertEqual((plain.x_pos, plain.y_pos, plain.orientation), (0.0, 0.0, 0.0))
        first = run(plain)
        second = run(explicit_zero)
        self.assertEqual(first.position_history, second.position_history)

    def test_start_pose_jitter_is_bounded_and_seeded(self):
        poses = set()
        for seed in range(30):
            vehicle = Vehicle(noise_seed=seed, start_position_stdev=0.2, start_heading_stdev=0.1)
            self.assertLessEqual(abs(vehicle.x_pos), 0.4)
            self.assertLessEqual(abs(vehicle.y_pos), 0.4)
            self.assertLessEqual(abs(vehicle.orientation), 0.2)
            poses.add((vehicle.x_pos, vehicle.y_pos, vehicle.orientation))
        self.assertEqual(len(poses), 30)
        again = Vehicle(noise_seed=3, start_position_stdev=0.2, start_heading_stdev=0.1)
        reference = Vehicle(noise_seed=3, start_position_stdev=0.2, start_heading_stdev=0.1)
        self.assertEqual((again.x_pos, again.orientation), (reference.x_pos, reference.orientation))

    def test_sensor_noise_perturbs_distances_within_range(self):
        maze = Maze(walls=[((3.0, -10.0), (3.0, 10.0))], obstacles=[], goal=(-8.0, -8.0))
        readings = []
        for seed in range(20):
            vehicle = Vehicle(noise_seed=seed, sensor_noise_stdev=0.2)
            vehicle.sense(maze)
            readings.append(vehicle.front_distance)
            self.assertTrue(0.0 <= vehicle.front_sensor <= 1.0)
        self.assertGreater(np.std(readings), 0.05)
        self.assertAlmostEqual(np.mean(readings), 2.0, delta=0.15)

    def test_motor_mismatch_curves_an_equal_command(self):
        straight = run(Vehicle(noise_stdev=0, controller=FixedController(0.5, 0.5),
                               navigation_heuristics=False))
        curved = run(Vehicle(noise_stdev=0, controller=FixedController(0.5, 0.5),
                             navigation_heuristics=False, motor_mismatch=0.05))
        self.assertAlmostEqual(straight.vehicle.orientation, 0.0)
        # orientation += turn_gain * (left - right); a stronger left wheel increases it
        self.assertGreater(curved.vehicle.orientation, 0.0)

    def test_motor_noise_varies_trajectories(self):
        ends = {
            run(Vehicle(noise_stdev=0, noise_seed=seed, controller=FixedController(0.5, 0.5),
                        navigation_heuristics=False, motor_noise_stdev=0.05)).vehicle.orientation
            for seed in range(5)
        }
        self.assertEqual(len(ends), 5)

    def test_evolution_worlds_use_default_noise(self):
        algorithm = GeneticAlgorithm(population_size=4, seed=1)
        controller = algorithm.decode_genome(algorithm.population[0].genome)
        first = algorithm._default_world_factory(controller).vehicle
        second = algorithm._default_world_factory(controller).vehicle
        for name in ("sensor_noise_stdev", "motor_noise_stdev", "motor_mismatch"):
            self.assertEqual(getattr(first, name), DEFAULT_ROBOT_NOISE[name])
        self.assertNotEqual((first.x_pos, first.orientation), (second.x_pos, second.orientation))

        quiet = GeneticAlgorithm(population_size=4, seed=1, robot_noise={})
        vehicle = quiet._default_world_factory(controller).vehicle
        self.assertEqual((vehicle.x_pos, vehicle.y_pos, vehicle.sensor_noise_stdev), (0.0, 0.0, 0.0))

        _, course_world = MultiCourseWorld(controller).worlds[0]
        self.assertEqual(course_world.vehicle.motor_mismatch, DEFAULT_ROBOT_NOISE["motor_mismatch"])


if __name__ == "__main__":
    unittest.main()
