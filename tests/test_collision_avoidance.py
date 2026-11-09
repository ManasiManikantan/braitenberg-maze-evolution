import unittest

from sim.maze import Maze
from sim.robot import Vehicle


class CollisionAvoidanceTests(unittest.TestCase):
    def test_clear_path_gets_forward_cruise_even_with_zero_controller(self):
        class ZeroController:
            @staticmethod
            def forward(sensors):
                return 0.0, 0.0

        maze = Maze(walls=[], obstacles=[], goal=(8.0, -7.0), goal_radius=1.0)
        vehicle = Vehicle(noise_stdev=0, controller=ZeroController())

        vehicle.sense(maze)
        vehicle.think()

        self.assertGreater(vehicle.left_motor, 0.0)
        self.assertGreater(vehicle.right_motor, 0.0)

    def test_front_obstacle_triggers_turn_in_place(self):
        maze = Maze(
            walls=[((4.0, -5.0), (4.0, 5.0))],
            obstacles=[],
            goal=(8.0, -7.0),
            goal_radius=1.0,
        )
        vehicle = Vehicle(noise_stdev=0, sensor_range=5, avoidance_distance=5)

        vehicle.sense(maze)
        vehicle.think()

        self.assertTrue(vehicle._turning_to_clear)
        self.assertAlmostEqual(vehicle.left_motor, -vehicle.right_motor)
        self.assertGreater(abs(vehicle.left_motor), 0.0)

    def test_turn_direction_moves_away_from_closer_side(self):
        vehicle = Vehicle(noise_stdev=0)
        vehicle.left_distance = 1.0
        vehicle.right_distance = 2.0

        vehicle.think()
        vehicle.move()

        self.assertLess(vehicle.orientation, 0.0)

    def test_turning_stops_and_forward_cruise_resumes_when_rays_clear(self):
        maze = Maze(walls=[], obstacles=[], goal=(0.0, 0.0), goal_radius=0.5)
        vehicle = Vehicle(noise_stdev=0)
        vehicle._turning_to_clear = True
        vehicle.left_distance = vehicle.right_distance = 1.0
        vehicle.think()

        self.assertTrue(vehicle._turning_to_clear)
        vehicle.left_distance = vehicle.sensor_range
        vehicle.right_distance = vehicle.avoidance_distance
        vehicle.think()

        self.assertTrue(vehicle._turning_to_clear)
        vehicle.right_distance = vehicle.sensor_range
        vehicle.left_sensor = vehicle.right_sensor = 0.0
        vehicle.think()

        self.assertFalse(vehicle._turning_to_clear)
        self.assertGreater(vehicle.left_motor, 0.0)
        self.assertGreater(vehicle.right_motor, 0.0)

    def test_collision_immediately_starts_avoidance_turn(self):
        maze = Maze(walls=[], obstacles=[], goal=(0.0, 0.0), goal_radius=0.5)
        vehicle = Vehicle(noise_stdev=0)
        vehicle.x_pos = 8.99
        vehicle.update_sensor_pos()
        vehicle.left_motor = vehicle.right_motor = 1.0

        self.assertTrue(vehicle.move(maze))
        self.assertTrue(vehicle._turning_to_clear)
        self.assertEqual(vehicle._reverse_steps_remaining, 0)


if __name__ == "__main__":
    unittest.main()