"""Maze geometry, obstacle raycasting, and robot collision checks."""

import math


class Maze:
	"""A rectangular maze with boundary walls, interior walls, and obstacles.

	Coordinates are centered at the origin. Walls are line segments represented
	by ``((x1, y1), (x2, y2))`` and obstacles are axis-aligned rectangles
	represented by ``(x_min, y_min, x_max, y_max)``.
	"""

	DEFAULT_WALLS = (
		((-5.0, -10.0), (-5.0, 4.0)),
		((-5.0, 4.0), (-2.0, 4.0)),
		((-2.0, 4.0), (-2.0, 8.0)),
		((1.0, -4.0), (5.0, -4.0)),
		((5.0, -4.0), (5.0, 10.0)),
		((5.0, 2.0), (10.0, 2.0)),
	)
	DEFAULT_OBSTACLES = (
		(1.5, 1.5, 2.5, 2.5),
		(7.0, -2.0, 8.0, -1.0),
	)
	CHALLENGE_WALLS = (
		((0.0, -8.0), (4.0, -8.0)),
		((2.0, -8.0), (2.0, -7.0)),
		((6.0, -5.5), (9.0, -5.5)),
	)
	CHALLENGE_OBSTACLES = (
		(-2.5, -3.5, -1.5, -2.5),
		(2.5, -3.5, 3.5, -2.5),
		(6.5, -3.5, 7.5, -2.5),
	)
	COURSE_THREE_WALLS = (
		((-10.0, 2.0), (3.5, 2.0)),
		((6.5, -10.0), (6.5, 1.0)),
	)
	COURSE_THREE_OBSTACLES = (
		(-3.0, -5.0, -2.0, -4.0),
		((1.0, 5.0, 2.0, 6.0)),
		((8.7, 4.5, 9.3, 5.2)),
	)
	COURSE_FOUR_WALLS = (
		((-3.0, -10.0), (-3.0, 6.0)),
		((-3.0, 6.0), (7.0, 6.0)),
		((7.0, 6.0), (7.0, -3.0)),
		((1.0, -4.0), (7.0, -4.0)),
	)
	COURSE_FOUR_OBSTACLES = (
		(-7.0, -7.0, -6.0, -6.0),
		(-1.0, 3.0, 0.0, 4.0),
		(3.0, 1.0, 4.0, 2.0),
		(8.0, 3.0, 9.0, 4.0),
	)

	def __init__(self, width=20.0, height=20.0, walls=None, obstacles=None,
			 goal=None, goal_radius=None, difficulty=1):
		if width <= 0 or height <= 0:
			raise ValueError("maze width and height must be positive")
		if difficulty not in (1, 2, 3, 4):
			raise ValueError("difficulty must be 1, 2, 3, or 4")

		self.width = float(width)
		self.height = float(height)
		self.difficulty = difficulty
		self.x_min = -self.width / 2
		self.x_max = self.width / 2
		self.y_min = -self.height / 2
		self.y_max = self.height / 2
		if walls is None:
			x_scale = self.width / 20.0
			y_scale = self.height / 20.0
			layout_walls = list(self.DEFAULT_WALLS)
			if difficulty == 2:
				layout_walls.extend(self.CHALLENGE_WALLS)
			elif difficulty == 3:
				layout_walls = list(self.COURSE_THREE_WALLS)
			elif difficulty == 4:
				layout_walls = list(self.COURSE_FOUR_WALLS)
			self.interior_walls = [
				((start[0] * x_scale, start[1] * y_scale),
				 (end[0] * x_scale, end[1] * y_scale))
				for start, end in layout_walls
			]
		else:
			self.interior_walls = list(walls)

		if obstacles is None:
			x_scale = self.width / 20.0
			y_scale = self.height / 20.0
			layout_obstacles = list(self.DEFAULT_OBSTACLES)
			if difficulty == 2:
				layout_obstacles.extend(self.CHALLENGE_OBSTACLES)
			elif difficulty == 3:
				layout_obstacles = list(self.COURSE_THREE_OBSTACLES)
			elif difficulty == 4:
				layout_obstacles = list(self.COURSE_FOUR_OBSTACLES)
			self.obstacles = [
				(x_min * x_scale, y_min * y_scale,
				 x_max * x_scale, y_max * y_scale)
				for x_min, y_min, x_max, y_max in layout_obstacles
			]
		else:
			self.obstacles = list(obstacles)
		self.walls = self._boundary_walls() + self.interior_walls

		for x_min, y_min, x_max, y_max in self.obstacles:
			if x_min >= x_max or y_min >= y_max:
				raise ValueError("obstacle bounds must have positive width and height")

		x_scale = self.width / 20.0
		y_scale = self.height / 20.0
		self.goal = (8.0 * x_scale, -7.0 * y_scale) if goal is None else tuple(goal)
		self.goal_radius = min(x_scale, y_scale) if goal_radius is None else goal_radius
		if self.goal_radius <= 0:
			raise ValueError("goal_radius must be positive")
		if self.collides(self.goal, self.goal_radius):
			raise ValueError("goal must not overlap a wall or obstacle")

	def _boundary_walls(self):
		return [
			((self.x_min, self.y_min), (self.x_max, self.y_min)),
			((self.x_max, self.y_min), (self.x_max, self.y_max)),
			((self.x_max, self.y_max), (self.x_min, self.y_max)),
			((self.x_min, self.y_max), (self.x_min, self.y_min)),
		]

	@staticmethod
	def _obstacle_walls(obstacle):
		x_min, y_min, x_max, y_max = obstacle
		return (
			((x_min, y_min), (x_max, y_min)),
			((x_max, y_min), (x_max, y_max)),
			((x_max, y_max), (x_min, y_max)),
			((x_min, y_max), (x_min, y_min)),
		)

	@staticmethod
	def _coordinates(point):
		if hasattr(point, "x_pos"):
			return point.x_pos, point.y_pos
		return point

	@staticmethod
	def _ray_segment_distance(origin, direction, segment):
		origin_x, origin_y = origin
		direction_x, direction_y = direction
		(start_x, start_y), (end_x, end_y) = segment
		segment_x = end_x - start_x
		segment_y = end_y - start_y
		offset_x = start_x - origin_x
		offset_y = start_y - origin_y
		denominator = direction_x * segment_y - direction_y * segment_x

		if denominator == 0:
			if offset_x * direction_y - offset_y * direction_x != 0:
				return None
			start_distance = offset_x * direction_x + offset_y * direction_y
			end_distance = ((end_x - origin_x) * direction_x +
							(end_y - origin_y) * direction_y)
			if max(start_distance, end_distance) < 0:
				return None
			return max(0.0, min(start_distance, end_distance))

		distance = (offset_x * segment_y - offset_y * segment_x) / denominator
		fraction = (offset_x * direction_y - offset_y * direction_x) / denominator
		if distance >= 0 and 0 <= fraction <= 1:
			return distance
		return None

	def raycast(self, origin, direction, max_distance):
		"""Return the nearest wall/obstacle distance, capped at ``max_distance``.

		``direction`` is normalized internally. If nothing is hit in range, the
		returned distance is ``max_distance``.
		"""
		return self.raycast_with_type(origin, direction, max_distance)[0]

	def raycast_with_type(self, origin, direction, max_distance):
		"""Return nearest distance and hit type (``wall``, ``obstacle``, or None)."""
		if max_distance <= 0:
			raise ValueError("max_distance must be positive")

		origin = self._coordinates(origin)
		direction_x, direction_y = direction
		magnitude = math.hypot(direction_x, direction_y)
		if magnitude == 0:
			raise ValueError("ray direction must be non-zero")
		direction = (direction_x / magnitude, direction_y / magnitude)

		nearest = float(max_distance)
		nearest_type = None
		for segment in self.walls:
			hit_distance = self._ray_segment_distance(origin, direction, segment)
			if hit_distance is not None and hit_distance < nearest:
				nearest = hit_distance
				nearest_type = "wall"
		for obstacle in self.obstacles:
			for segment in self._obstacle_walls(obstacle):
				hit_distance = self._ray_segment_distance(origin, direction, segment)
				if hit_distance is not None and hit_distance < nearest:
					nearest = hit_distance
					nearest_type = "obstacle"
		return nearest, nearest_type

	@staticmethod
	def _point_segment_distance_squared(point, segment):
		point_x, point_y = point
		(start_x, start_y), (end_x, end_y) = segment
		segment_x = end_x - start_x
		segment_y = end_y - start_y
		length_squared = segment_x**2 + segment_y**2
		if length_squared == 0:
			return (point_x - start_x)**2 + (point_y - start_y)**2

		projection = ((point_x - start_x) * segment_x +
					  (point_y - start_y) * segment_y) / length_squared
		projection = max(0.0, min(1.0, projection))
		closest_x = start_x + projection * segment_x
		closest_y = start_y + projection * segment_y
		return (point_x - closest_x)**2 + (point_y - closest_y)**2

	def collides(self, position, radius):
		"""Return whether a circular robot body overlaps any wall or obstacle."""
		if radius < 0:
			raise ValueError("radius must be non-negative")
		x_pos, y_pos = self._coordinates(position)
		if (x_pos - radius < self.x_min or x_pos + radius > self.x_max or
				y_pos - radius < self.y_min or y_pos + radius > self.y_max):
			return True

		radius_squared = radius**2
		if any(self._point_segment_distance_squared((x_pos, y_pos), wall) <= radius_squared
			   for wall in self.interior_walls):
			return True

		for obstacle in self.obstacles:
			x_min, y_min, x_max, y_max = obstacle
			nearest_x = max(x_min, min(x_pos, x_max))
			nearest_y = max(y_min, min(y_pos, y_max))
			if (x_pos - nearest_x)**2 + (y_pos - nearest_y)**2 <= radius_squared:
				return True
		return False

	def clearance(self, position, radius=0.0):
		"""Return signed distance from a circular body to the nearest solid.

		Positive values are free space, zero is contact, and negative values mean
		the body overlaps a wall or obstacle.
		"""
		if radius < 0:
			raise ValueError("radius must be non-negative")
		x_pos, y_pos = self._coordinates(position)
		clearances = [
			x_pos - self.x_min,
			self.x_max - x_pos,
			y_pos - self.y_min,
			self.y_max - y_pos,
		]
		clearances.extend(
			math.sqrt(self._point_segment_distance_squared((x_pos, y_pos), wall))
			for wall in self.interior_walls
		)
		for obstacle in self.obstacles:
			x_min, y_min, x_max, y_max = obstacle
			if x_min <= x_pos <= x_max and y_min <= y_pos <= y_max:
				clearances.append(-min(
					x_pos - x_min, x_max - x_pos,
					y_pos - y_min, y_max - y_pos,
				))
			else:
				nearest_x = max(x_min, min(x_pos, x_max))
				nearest_y = max(y_min, min(y_pos, y_max))
				clearances.append(math.hypot(x_pos - nearest_x, y_pos - nearest_y))
		return min(clearances) - radius


__all__ = ["Maze"]
