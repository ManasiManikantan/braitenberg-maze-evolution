"""Step a vehicle through a maze environment."""

import math

from .maze import Maze
from .robot import Vehicle


class World:
	"""Coordinate maze sensing, controller decisions, and vehicle motion."""

	def __init__(self, maze=None, vehicle=None, goal=None, goal_radius=None):
		self.maze = Maze() if maze is None else maze
		self.vehicle = Vehicle() if vehicle is None else vehicle
		self.goal = self.maze.goal if goal is None else self._coordinates(goal)
		self.goal_radius = self.maze.goal_radius if goal_radius is None else goal_radius
		if self.goal_radius <= 0:
			raise ValueError("goal_radius must be positive")
		self.steps = 0
		self.collisions = 0
		self.moving_steps = 0
		self.position_history = [(self.vehicle.x_pos, self.vehicle.y_pos)]
		self.initial_distance = self._distance_to_goal(self.position_history[0])
		self.reached_goal = self.initial_distance <= self.goal_radius

	@staticmethod
	def _coordinates(point):
		if hasattr(point, "x_pos"):
			return point.x_pos, point.y_pos
		return tuple(point)

	def _distance_to_goal(self, position):
		position_x, position_y = self._coordinates(position)
		return math.hypot(position_x - self.goal[0], position_y - self.goal[1])

	def step(self):
		"""Advance the simulation by one sense-think-move cycle."""
		if self.reached_goal:
			return self.vehicle
		old_position = (self.vehicle.x_pos, self.vehicle.y_pos)
		self.vehicle.sense(self.maze)
		self.vehicle.think()
		if self.vehicle.move(self.maze):
			self.collisions += 1
		new_position = (self.vehicle.x_pos, self.vehicle.y_pos)
		if math.hypot(new_position[0] - old_position[0],
					  new_position[1] - old_position[1]) > 1e-8:
			self.moving_steps += 1
		self.position_history.append(new_position)
		self.steps += 1
		self.reached_goal = self._distance_to_goal(new_position) <= self.goal_radius
		return self.vehicle

	def run(self, steps):
		"""Advance the simulation for ``steps`` cycles and return the vehicle."""
		if steps < 0:
			raise ValueError("steps must be non-negative")
		for _ in range(steps):
			if self.reached_goal:
				break
			self.step()
		return self.vehicle
