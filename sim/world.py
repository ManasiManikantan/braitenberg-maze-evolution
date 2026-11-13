"""Step a vehicle through a maze environment."""

import math

from .maze import Maze
from .robot import Vehicle


class World:
	"""Coordinate maze sensing, controller decisions, and vehicle motion."""

	def __init__(self, maze=None, vehicle=None, goal=None, goal_radius=None,
			 goal_tolerance=None, revisit_cell_size=1.0, revisit_trigger=2,
			 stagnation_limit=80, progress_tolerance=0.25, health=None, damage_per_hit=1.0,
			 energy=None, contact_step_cost=0.0):
		"""``health`` (None = off) turns on "walls hurt": every new collision
		(entering contact, not every step spent pinned against a wall) costs
		``damage_per_hit``, and every step spent in contact (including the
		first) costs ``contact_step_cost`` (default 0); at 0 the robot is
		``dead`` and the run ends without reaching the goal. ``contact_entries`` counts new collisions either way.

		``energy`` (None = off) turns on the battery: every step costs
		|left motor| + |right motor| (the clipped commands); when it reaches 0
		the robot is ``out_of_energy`` and the run ends without the goal.
		``stopped`` is True once the robot is dead or out of energy.
		"""
		self.maze = Maze() if maze is None else maze
		self.vehicle = Vehicle() if vehicle is None else vehicle
		self.goal = self.maze.goal if goal is None else self._coordinates(goal)
		self.goal_radius = self.maze.goal_radius if goal_radius is None else goal_radius
		if self.goal_radius <= 0:
			raise ValueError("goal_radius must be positive")
		default_tolerance = self.goal_radius + self.vehicle.radius
		self.goal_tolerance = (
			default_tolerance if goal_tolerance is None else goal_tolerance
		)
		if self.goal_tolerance <= 0:
			raise ValueError("goal_tolerance must be positive")
		if revisit_cell_size <= 0:
			raise ValueError("revisit_cell_size must be positive")
		if revisit_trigger < 2:
			raise ValueError("revisit_trigger must be at least 2")
		if stagnation_limit < 1:
			raise ValueError("stagnation_limit must be at least 1")
		if progress_tolerance < 0:
			raise ValueError("progress_tolerance must be non-negative")
		if health is not None and health <= 0:
			raise ValueError("health must be positive or None")
		if damage_per_hit < 0 or contact_step_cost < 0:
			raise ValueError("damage_per_hit and contact_step_cost must be non-negative")
		self.contact_step_cost = contact_step_cost
		self.health = health
		self.damage_per_hit = damage_per_hit
		self.dead = False
		if energy is not None and energy <= 0:
			raise ValueError("energy must be positive or None")
		self.energy = energy
		self.out_of_energy = False
		self.energy_used = 0.0
		self.contact_entries = 0
		self._in_contact = False
		self.steps = 0
		self.collisions = 0
		self.moving_steps = 0
		self.position_history = [(self.vehicle.x_pos, self.vehicle.y_pos)]
		self.revisit_cell_size = revisit_cell_size
		self.revisit_trigger = revisit_trigger
		self.path_visit_counts = {}
		self.path_revisits = 0
		self._last_cell = self._path_cell(self.position_history[0])
		self.path_visit_counts[self._last_cell] = 1
		self.initial_distance = self._distance_to_goal(self.position_history[0])
		self._best_distance_since_progress = self.initial_distance
		self.stagnant_steps = 0
		self.stagnation_limit = stagnation_limit
		self.progress_tolerance = progress_tolerance
		self.reached_goal = self.initial_distance <= self.goal_tolerance

	@staticmethod
	def _coordinates(point):
		if hasattr(point, "x_pos"):
			return point.x_pos, point.y_pos
		return tuple(point)

	def _distance_to_goal(self, position):
		position_x, position_y = self._coordinates(position)
		return math.hypot(position_x - self.goal[0], position_y - self.goal[1])

	def _path_cell(self, position):
		position_x, position_y = self._coordinates(position)
		column = int((position_x - self.maze.x_min) / self.revisit_cell_size)
		row = int((position_y - self.maze.y_min) / self.revisit_cell_size)
		return column, row

	def _record_path_visit(self, position):
		cell = self._path_cell(position)
		if cell == self._last_cell:
			return False

		visits = self.path_visit_counts.get(cell, 0) + 1
		self.path_visit_counts[cell] = visits
		self._last_cell = cell
		if visits >= self.revisit_trigger:
			self.path_revisits += 1
			self.vehicle.request_avoidance_scan()
			return True
		return False

	def step(self):
		"""Advance the simulation by one sense-think-move cycle."""
		if self.reached_goal or self.stopped:
			return self.vehicle
		old_position = (self.vehicle.x_pos, self.vehicle.y_pos)
		self.vehicle.sense(self.maze, self.goal)
		self.vehicle.think()
		collided = self.vehicle.move(self.maze)
		if collided:
			self.collisions += 1
			if not self._in_contact:
				self.contact_entries += 1
			if self.health is not None:
				self.health -= self.contact_step_cost
				if not self._in_contact:
					self.health -= self.damage_per_hit
				self.dead = self.health <= 0
		self._in_contact = bool(collided)
		cost = abs(self.vehicle.left_motor) + abs(self.vehicle.right_motor)
		self.energy_used += cost
		if self.energy is not None:
			self.energy -= cost
			self.out_of_energy = self.energy <= 0
		new_position = (self.vehicle.x_pos, self.vehicle.y_pos)
		if math.hypot(new_position[0] - old_position[0],
					  new_position[1] - old_position[1]) > 1e-8:
			self.moving_steps += 1
		self.position_history.append(new_position)
		self.steps += 1
		goal_distance = self._distance_to_goal(new_position)
		self.reached_goal = goal_distance <= self.goal_tolerance
		if not self.reached_goal:
			self._record_path_visit(new_position)
			if self._best_distance_since_progress - goal_distance >= self.progress_tolerance:
				self._best_distance_since_progress = goal_distance
				self.stagnant_steps = 0
			else:
				self.stagnant_steps += 1
			if self.stagnant_steps >= self.stagnation_limit:
				self.vehicle.request_avoidance_scan()
				self._best_distance_since_progress = goal_distance
				self.stagnant_steps = 0
		return self.vehicle

	@property
	def stopped(self):
		"""True once the run has ended early (dead or out of energy)."""
		return self.dead or self.out_of_energy

	def run(self, steps):
		"""Advance the simulation for ``steps`` cycles and return the vehicle."""
		if steps < 0:
			raise ValueError("steps must be non-negative")
		for _ in range(steps):
			if self.reached_goal or self.stopped:
				break
			self.step()
		return self.vehicle
