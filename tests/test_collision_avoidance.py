import math
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
        vehicle.left_distance = 6.0
        vehicle.right_distance = 1.0

        vehicle.think()
        vehicle.move()

        self.assertFalse(vehicle._turning_to_clear)
        self.assertLess(vehicle.orientation, 0.0)
        self.assertGreater(vehicle.velocity, 0.0)

    def test_forward_half_scan_records_best_heading_then_aligns(self):
        vehicle = Vehicle(noise_stdev=0)
        vehicle.left_distance = vehicle.right_distance = 1.0
        vehicle._begin_avoidance_turn()
        expected_heading = None
        for _ in range(20):
            vehicle.think()
            vehicle.move()
            if vehicle._scan_phase == "sweep":
                break

        self.assertEqual(vehicle._scan_phase, "sweep")
        for step in range(21):
            if step == 10:
                vehicle.left_distance = 9.0
                vehicle.right_distance = 4.0
                expected_heading = vehicle.orientation
            else:
                vehicle.left_distance = vehicle.right_distance = 4.0
            vehicle.think()
            vehicle.move()

        vehicle.left_distance = vehicle.right_distance = 5.0
        vehicle.think()

        self.assertFalse(vehicle._turning_to_clear)
        self.assertTrue(vehicle._aligning_to_opening)
        self.assertAlmostEqual(vehicle._opening_heading, expected_heading)
        self.assertLessEqual(abs(expected_heading), math.pi / 2)
        self.assertAlmostEqual(vehicle._scan_rotation, math.pi, delta=0.04)

        for _ in range(12):
            vehicle.move()
            vehicle.think()
            if not vehicle._aligning_to_opening:
                break

        self.assertFalse(vehicle._aligning_to_opening)
        heading_error = math.atan2(
            math.sin(expected_heading - vehicle.orientation),
            math.cos(expected_heading - vehicle.orientation),
        )
        self.assertLess(abs(heading_error), 0.08)
        self.assertGreater(vehicle.left_motor, 0.0)
        self.assertGreater(vehicle.right_motor, 0.0)

    def test_robot_moves_toward_sensor_with_longer_range(self):
        vehicle = Vehicle(noise_stdev=0)
        vehicle.left_distance = 6.0
        vehicle.right_distance = 1.0

        vehicle.think()
        vehicle.move()

        self.assertFalse(vehicle._turning_to_clear)
        self.assertGreater(vehicle.orientation, 0.0)

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