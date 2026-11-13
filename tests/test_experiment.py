import unittest
import math

from controller.network import NeuralNetwork
from evolution.experiment import (
    MultiCourseWorld,
    course_step_limit,
    evaluate_controller,
    make_legacy_courses,
)
from sim.maze import Maze
from sim.robot import Vehicle


class FixedController:
    def forward(self, sensors):
        return (0.8, 0.4)


class ExperimentTests(unittest.TestCase):
    def test_evaluation_set_contains_training_and_held_out_courses(self):
        courses = make_legacy_courses()

        self.assertEqual(len(courses), 6)
        self.assertEqual(
            [group for _, group, _ in courses],
            ["training"] * 4 + ["held_out"] * 2,
        )
        self.assertEqual(
            len({(tuple(maze.interior_walls), tuple(maze.obstacles)) for _, _, maze in courses}),
            6,
        )
        for _, _, maze in courses:
            self.assertFalse(maze.collides(maze.goal, maze.goal_radius))

    def test_held_out_courses_have_body_clear_routes_to_goal(self):
        courses = make_legacy_courses()[4:]
        routes = (
            [(0.0, 0.0), (0.0, 4.0), (3.5, 4.0), (3.5, 0.0),
             (6.5, 0.0), (6.5, -7.0), (8.0, -7.0)],
            [(0.0, 0.0), (0.0, -3.75), (6.5, -3.75), (6.5, -7.0), (8.0, -7.0)],
        )
        for (_, _, maze), route in zip(courses, routes):
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

    def test_evaluation_records_requested_comparison_metrics(self):
        course = make_legacy_courses()[:1]
        records = evaluate_controller(
            NeuralNetwork(seed=4),
            "random_init",
            "full",
            training_seed=2,
            courses=course,
            steps=0,
            noise_stdev=0.05,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["goal_success"], 0)
        self.assertEqual(records[0]["collision_count"], 0)
        self.assertEqual(records[0]["completion_steps"], "")
        self.assertEqual(records[0]["evaluation_steps"], 0)
        self.assertEqual(records[0]["noise_seed"], 20000)
        self.assertIn("path_length", records[0])
        self.assertIn("fitness", records[0])

    def test_neural_only_mode_bypasses_goal_seeking(self):
        maze = Maze(walls=[], obstacles=[], goal=(3.0, 0.0), goal_radius=1.0)
        vehicle = Vehicle(
            controller=FixedController(),
            noise_stdev=0,
            navigation_heuristics=False,
        )
        vehicle.sense(maze, maze.goal)
        vehicle.think()

        self.assertEqual(vehicle.left_motor, 0.8)
        self.assertEqual(vehicle.right_motor, 0.4)
        vehicle.request_avoidance_scan()
        self.assertFalse(vehicle._turning_to_clear)

    def test_noise_seed_makes_movement_reproducible(self):
        vehicles = [
            Vehicle(noise_stdev=0.1, noise_seed=17)
            for _ in range(2)
        ]
        for vehicle in vehicles:
            vehicle.left_motor = 0.5
            vehicle.right_motor = 0.25
            vehicle.move()

        self.assertEqual(vehicles[0].orientation, vehicles[1].orientation)
        self.assertEqual(vehicles[0].x_pos, vehicles[1].x_pos)
        self.assertEqual(vehicles[0].y_pos, vehicles[1].y_pos)


class StandStill:
    def forward(self, sensors):
        return (0.0, 0.0)


class TimeLimitTests(unittest.TestCase):
    def test_limit_is_factor_times_shortest_path_driving_time(self):
        maze = Maze(difficulty=3)
        self.assertEqual(course_step_limit(maze, 2.0, 0.05), 740)
        self.assertEqual(course_step_limit(maze, 1.0, 0.05), 370)
        self.assertEqual(course_step_limit(maze, 2.0, 0.1), 370)
        self.assertIsNone(course_step_limit(maze, 0))
        with self.assertRaises(ValueError):
            course_step_limit(maze, 2.0, 0.0)

    def test_training_worlds_stop_at_each_course_limit(self):
        limited = MultiCourseWorld(StandStill(), time_limit_factor=2.0)
        limited.run(3000)
        self.assertEqual([world.steps for _, world in limited.worlds], [600, 600, 740, 600])

        unlimited = MultiCourseWorld(StandStill(), time_limit_factor=0)
        unlimited.run(1000)
        self.assertEqual([world.steps for _, world in unlimited.worlds], [1000] * 4)

        capped = MultiCourseWorld(StandStill(), time_limit_factor=2.0)
        capped.run(500)
        self.assertEqual([world.steps for _, world in capped.worlds], [500] * 4)

    def test_evaluation_uses_per_course_limits(self):
        courses = [("course_3", "training", Maze(difficulty=3))]
        record, = evaluate_controller(StandStill(), "still", "neural_only", 0, courses, 3000, 0.0,
                                      time_limit_factor=2.0)
        self.assertEqual(record["evaluation_steps"], 740)
        record, = evaluate_controller(StandStill(), "still", "neural_only", 0, courses, 3000, 0.0,
                                      time_limit_factor=0)
        self.assertEqual(record["evaluation_steps"], 3000)


if __name__ == "__main__":
    unittest.main()
