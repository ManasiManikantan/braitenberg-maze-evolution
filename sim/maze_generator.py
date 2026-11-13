"""Random perfect (loop-free) mazes for training controllers on varied layouts."""

from collections import deque

import numpy as np

from .maze import Maze


class RandomMaze:
	"""A generated maze plus the start pose and route length chosen for it."""

	def __init__(self, maze, start, start_heading, route_cells, mirrored):
		self.maze = maze
		self.start = start
		self.start_heading = start_heading
		self.route_cells = route_cells
		self.mirrored = mirrored


def _carve_spanning_tree(rng, columns, rows):
	"""Return the set of open passages of a random depth-first spanning tree."""
	visited = np.zeros((columns, rows), dtype=bool)
	start = (int(rng.integers(columns)), int(rng.integers(rows)))
	stack = [start]
	visited[start] = True
	passages = set()
	while stack:
		column, row = stack[-1]
		neighbours = [
			(column + d_column, row + d_row)
			for d_column, d_row in ((1, 0), (-1, 0), (0, 1), (0, -1))
			if 0 <= column + d_column < columns and 0 <= row + d_row < rows
			and not visited[column + d_column, row + d_row]
		]
		if not neighbours:
			stack.pop()
			continue
		chosen = neighbours[int(rng.integers(len(neighbours)))]
		visited[chosen] = True
		passages.add(frozenset(((column, row), chosen)))
		stack.append(chosen)
	return passages


def _route_lengths(passages, columns, rows, source):
	"""Breadth-first route length (in cells) from ``source`` to every cell."""
	lengths = {source: 0}
	queue = deque([source])
	while queue:
		column, row = queue.popleft()
		for d_column, d_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
			neighbour = (column + d_column, row + d_row)
			if neighbour not in lengths and frozenset(((column, row), neighbour)) in passages:
				lengths[neighbour] = lengths[(column, row)] + 1
				queue.append(neighbour)
	return lengths


def _interior_walls(passages, columns, rows, cell_size, x_min, y_min):
	"""Wall segments between adjacent cells that are not connected, merged into runs."""
	walls = []
	# vertical walls: between (column, row) and (column + 1, row)
	for column in range(columns - 1):
		x_pos = x_min + (column + 1) * cell_size
		run_start = None
		for row in range(rows + 1):
			closed = row < rows and frozenset(((column, row), (column + 1, row))) not in passages
			if closed and run_start is None:
				run_start = row
			elif not closed and run_start is not None:
				walls.append(((x_pos, y_min + run_start * cell_size), (x_pos, y_min + row * cell_size)))
				run_start = None
	# horizontal walls: between (column, row) and (column, row + 1)
	for row in range(rows - 1):
		y_pos = y_min + (row + 1) * cell_size
		run_start = None
		for column in range(columns + 1):
			closed = column < columns and frozenset(((column, row), (column, row + 1))) not in passages
			if closed and run_start is None:
				run_start = column
			elif not closed and run_start is not None:
				walls.append(((x_min + run_start * cell_size, y_pos), (x_min + column * cell_size, y_pos)))
				run_start = None
	return walls


def maze_from_passages(passages, columns, rows, cell_size, goal_cell, start_cell,
					   goal_radius=1.0):
	"""Build a centred grid maze from open passages between adjacent cells.

	``passages`` is an iterable of ``(cell, cell)`` pairs; every other edge
	between adjacent cells is a wall. Returns ``(maze, start_position)`` with
	the goal and start at the centres of ``goal_cell`` and ``start_cell``.
	"""
	passages = {frozenset(pair) for pair in passages}
	for pair in passages:
		(c1, r1), (c2, r2) = tuple(pair)
		if abs(c1 - c2) + abs(r1 - r2) != 1:
			raise ValueError("passages must join adjacent cells")
	x_min = -columns * cell_size / 2
	y_min = -rows * cell_size / 2

	def centre(cell):
		return (x_min + (cell[0] + 0.5) * cell_size, y_min + (cell[1] + 0.5) * cell_size)

	walls = _interior_walls(passages, columns, rows, cell_size, x_min, y_min)
	maze = Maze(width=columns * cell_size, height=rows * cell_size, walls=walls, obstacles=[],
				goal=centre(goal_cell), goal_radius=goal_radius)
	return maze, centre(start_cell)


def generate_perfect_maze(
	rng,
	size=20.0,
	cell_size=4.0,
	min_route_cells=3,
	max_route_cells=6,
	mirror_probability=0.0,
	goal_radius=1.0,
):
	"""Generate a random loop-free maze with a random start pose and goal.

	The ``size`` x ``size`` area (centred on the origin like the built-in
	courses) is divided into ``cell_size`` squares and carved into a perfect
	maze by randomized depth-first search, so every pair of cells is joined
	by exactly one route. Corridors are ``cell_size`` wide (default 4, twice
	the robot's diameter). The goal and start are the centres of two cells
	whose route is ``min_route_cells`` to ``max_route_cells`` cells long; the
	start heading is uniform in [0, 2*pi). With probability
	``mirror_probability`` the layout is mirrored left-right.
	"""
	if cell_size <= 0 or size < cell_size:
		raise ValueError("cell_size must be positive and no larger than size")
	columns = rows = int(round(size / cell_size))
	if columns < 2:
		raise ValueError("the maze must be at least two cells across")
	if not 1 <= min_route_cells <= max_route_cells:
		raise ValueError("route cell bounds must satisfy 1 <= min <= max")
	if not 0.0 <= mirror_probability <= 1.0:
		raise ValueError("mirror_probability must be between 0 and 1")
	if 2 * goal_radius >= cell_size:
		raise ValueError("goal_radius must fit inside a cell")

	half = columns * cell_size / 2
	x_min = y_min = -half
	while True:
		passages = _carve_spanning_tree(rng, columns, rows)
		goal_cell = (int(rng.integers(columns)), int(rng.integers(rows)))
		lengths = _route_lengths(passages, columns, rows, goal_cell)
		candidates = sorted(
			cell for cell, length in lengths.items()
			if min_route_cells <= length <= max_route_cells
		)
		if candidates:
			break
	start_cell = candidates[int(rng.integers(len(candidates)))]

	def centre(cell):
		return (x_min + (cell[0] + 0.5) * cell_size, y_min + (cell[1] + 0.5) * cell_size)

	walls = _interior_walls(passages, columns, rows, cell_size, x_min, y_min)
	goal = centre(goal_cell)
	start = centre(start_cell)
	heading = float(rng.uniform(0.0, 2 * np.pi))
	mirrored = bool(rng.random() < mirror_probability)
	if mirrored:
		walls = [((-x1, y1), (-x2, y2)) for (x1, y1), (x2, y2) in walls]
		goal = (-goal[0], goal[1])
		start = (-start[0], start[1])
		heading = float((np.pi - heading) % (2 * np.pi))

	maze = Maze(width=2 * half, height=2 * half, walls=walls, obstacles=[],
				goal=goal, goal_radius=goal_radius)
	return RandomMaze(maze, start, heading, lengths[start_cell], mirrored)


__all__ = ["RandomMaze", "generate_perfect_maze", "maze_from_passages"]


def dead_end_cells(maze, cell_size=4.0):
	"""Centres of the cells with exactly one open side (walls on three sides)."""
	columns = int(round(maze.width / cell_size))
	rows = int(round(maze.height / cell_size))
	ends = []
	for column in range(columns):
		for row in range(rows):
			centre = (maze.x_min + (column + 0.5) * cell_size, maze.y_min + (row + 0.5) * cell_size)
			open_sides = sum(
				maze.raycast(centre, direction, cell_size) >= cell_size - 1e-6
				for direction in ((1, 0), (-1, 0), (0, 1), (0, -1))
			)
			if open_sides == 1:
				ends.append(centre)
	return ends


def with_goal_in_dead_end(random_maze, rng, cell_size=4.0):
	"""Copy of ``random_maze`` with the goal moved to a random dead end (not the start cell)."""
	maze = random_maze.maze
	start = random_maze.start
	ends = [c for c in dead_end_cells(maze, cell_size)
			if abs(c[0] - start[0]) > 1e-6 or abs(c[1] - start[1]) > 1e-6]
	goal = ends[int(rng.integers(len(ends)))]
	moved = Maze(width=maze.width, height=maze.height, walls=maze.interior_walls, obstacles=[],
				 goal=goal, goal_radius=maze.goal_radius)
	return RandomMaze(moved, start, random_maze.start_heading, None, random_maze.mirrored)

