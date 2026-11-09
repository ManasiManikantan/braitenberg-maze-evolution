import math
import unittest

from sim.maze import Maze
from sim.robot import Vehicle
from sim.world import World


class CollisionAvoidanceTests(unittest.TestCase):
    def test_challenge_maze_adds_barriers_and_obstacles(self):
        standard = Maze()
        challenge = Maze(difficulty=2)

        self.assertGreater(len(challenge.interior_walls), len(standard.interior_walls))
        self.assertGreater(len(challenge.obstacles), len(standard.obstacles))
        self.assertFalse(challenge.collides(challenge.goal, challenge.goal_radius))

    def test_challenge_maze_has_body_clear_route_to_goal(self):
        maze = Maze(difficulty=2)
        route = [
            (0.0, 0.0),
            (-0.5, -5.5),
            (4.8, -5.5),
            (4.8, -6.8),
            (8.0, -6.8),
            maze.goal,
        ]

        for start, end in zip(route, route[1:]):
            distance = math.hypot(end[0] - start[0], end[1] - start[1])
            samples = max(1, int(distance / 0.1))
            for sample in range(samples + 1):
                fraction = sample / samples
                position = (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
                self.assertFalse(maze.collides(position, radius=1.0), position)

    def test_challenge_maze_has_body_clear_route_to_goal(self):
        maze = Maze(difficulty=2)
        route = [
            (0.0, 0.0),
            (-0.5, -5.5),
            (4.8, -5.5),
            (4.8, -6.8),
            (8.0, -6.8),
            maze.goal,
        ]

        for start, end in zip(route, route[1:]):
            distance = math.hypot(end[0] - start[0], end[1] - start[1])
            samples = max(1, int(distance / 0.1))
            for sample in range(samples + 1):
                fraction = sample / samples
                position = (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
                self.assertFalse(maze.collides(position, radius=1.0), position)

    def test_clear_path_gets_forward_cruise_even_if_controller_reverses(self):
        class ReverseController:
            @staticmethod
            def forward(sensors):
                return -1.0, -1.0

        maze = Maze(walls=[], obstacles=[], goal=(8.0, -7.0), goal_radius=1.0)
        vehicle = Vehicle(noise_stdev=0, controller=ReverseController())

        vehicle.sense(maze)
        vehicle.think()

        self.assertGreater(vehicle.left_motor, 0.0)
        self.assertGreater(vehicle.right_motor, 0.0)

    def test_robot_steers_toward_visible_goal(self):
        class ZeroController:
            @staticmethod
            def forward(sensors):
                return 0.0, 0.0

        maze = Maze(walls=[], obstacles=[], goal=(0.0, 3.0), goal_radius=1.0)
        vehicle = Vehicle(noise_stdev=0, controller=ZeroController())
        world = World(maze=maze, vehicle=vehicle)

        world.step()

        self.assertTrue(vehicle.goal_visible)
        self.assertGreater(vehicle.orientation, 0.0)
        self.assertGreater(vehicle.y_pos, 0.0)

    def test_distant_visible_goal_does_not_override_navigation(self):
        class ZeroController:
            @staticmethod
            def forward(sensors):
                return 0.0, 0.0

        maze = Maze(walls=[], obstacles=[], goal=(0.0, 8.0), goal_radius=1.0)
        vehicle = Vehicle(noise_stdev=0, controller=ZeroController())
        world = World(maze=maze, vehicle=vehicle)

        world.step()

        self.assertTrue(vehicle.goal_visible)
        self.assertGreater(vehicle._goal_distance, vehicle.goal_seek_distance)
        self.assertAlmostEqual(vehicle.orientation, 0.0)
        self.assertGreater(vehicle.y_pos, 0.0)

    def test_wall_blocks_goal_visibility(self):
        maze = Maze(
            walls=[((-5.0, 4.0), (5.0, 4.0))],
            obstacles=[],
            goal=(0.0, 8.0),
            goal_radius=1.0,
        )
        vehicle = Vehicle(noise_stdev=0)

        vehicle.sense(maze, maze.goal)

        self.assertFalse(vehicle.goal_visible)

    def test_world_reaches_goal_when_robot_body_overlaps_it(self):
        maze = Maze(walls=[], obstacles=[], goal=(0.0, 5.0), goal_radius=1.0)
        overlapping_robot = Vehicle(noise_stdev=0)
        overlapping_robot.y_pos = 3.1
        separated_robot = Vehicle(noise_stdev=0)
        separated_robot.y_pos = 2.9

        overlapping_world = World(maze=maze, vehicle=overlapping_robot)
        separated_world = World(maze=maze, vehicle=separated_robot)

        self.assertTrue(overlapping_world.reached_goal)
        self.assertFalse(separated_world.reached_goal)

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

    def test_center_sensor_detects_wall_ahead(self):
        maze = Maze(
            walls=[((4.0, -5.0), (4.0, 5.0))],
            obstacles=[],
            goal=(8.0, -7.0),
            goal_radius=1.0,
        )
        vehicle = Vehicle(noise_stdev=0)

        vehicle.sense(maze)

        self.assertAlmostEqual(vehicle.front_distance, 3.0)
        self.assertAlmostEqual(vehicle.left_distance, 4.0)
        self.assertAlmostEqual(vehicle.right_distance, 4.0)
        self.assertLess(vehicle.front_distance, vehicle.left_distance)
        self.assertLess(vehicle.front_distance, vehicle.right_distance)

    def test_forward_half_scan_prefers_wide_body_clearance(self):
        vehicle = Vehicle(noise_stdev=0)
        vehicle.left_distance = vehicle.right_distance = 1.0
        vehicle._begin_avoidance_turn()
        best_heading = None

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
            elif step == 15:
                vehicle.left_distance = vehicle.right_distance = 6.0
                best_heading = vehicle.orientation
            else:
                vehicle.left_distance = vehicle.right_distance = 4.0
            vehicle.think()
            vehicle.move()

        vehicle.left_distance = vehicle.right_distance = 5.0
        vehicle.think()

        self.assertFalse(vehicle._turning_to_clear)
        self.assertTrue(vehicle._aligning_to_opening)
        self.assertAlmostEqual(vehicle._opening_heading, best_heading)
        self.assertLessEqual(abs(best_heading), math.pi / 2)
        self.assertAlmostEqual(vehicle._scan_rotation, math.pi, delta=0.04)

        for _ in range(12):
            vehicle.move()
            vehicle.think()
            if not vehicle._aligning_to_opening:
                break

        self.assertFalse(vehicle._aligning_to_opening)
        heading_error = math.atan2(
            math.sin(best_heading - vehicle.orientation),
            math.cos(best_heading - vehicle.orientation),
        )
        self.assertLess(abs(heading_error), 0.08)
        self.assertGreater(vehicle.left_motor, 0.0)
        self.assertGreater(vehicle.right_motor, 0.0)

    def test_robot_moves_toward_sensor_with_longer_clearance(self):
        vehicle = Vehicle(noise_stdev=0)
        vehicle.left_distance = 6.0
        vehicle.right_distance = 1.0

        vehicle.think()
        vehicle.move()

        self.assertFalse(vehicle._turning_to_clear)
        self.assertGreater(vehicle.orientation, 0.0)
        self.assertGreater(vehicle.velocity, 0.0)

    def test_collision_starts_short_reverse(self):
        maze = Maze(walls=[], obstacles=[], goal=(0.0, 0.0), goal_radius=0.5)
        vehicle = Vehicle(noise_stdev=0)
        vehicle.x_pos = 8.99
        vehicle.update_sensor_pos()
        vehicle.left_motor = vehicle.right_motor = 1.0

        self.assertTrue(vehicle.move(maze))
        self.assertFalse(vehicle._turning_to_clear)
        self.assertEqual(
            vehicle._reverse_steps_remaining,
            vehicle.collision_reverse_steps,
        )

        vehicle.think()
        self.assertLess(vehicle.left_motor, 0.0)
        self.assertLess(vehicle.right_motor, 0.0)
        self.assertFalse(vehicle.move(maze))
        self.assertLess(vehicle.x_pos, 8.99)

    def test_collision_during_reverse_starts_scan(self):
        maze = Maze(
            width=2.01,
            height=10.0,
            walls=[],
            obstacles=[],
            goal=(0.0, 4.0),
            goal_radius=0.1,
        )
        vehicle = Vehicle(noise_stdev=0)
        vehicle.left_motor = vehicle.right_motor = 1.0

        self.assertTrue(vehicle.move(maze))
        vehicle.think()

        self.assertTrue(vehicle.move(maze))
        self.assertEqual(vehicle._reverse_steps_remaining, 0)
        self.assertTrue(vehicle._turning_to_clear)


if __name__ == "__main__":
    unittest.main()
