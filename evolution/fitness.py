"""Fitness evaluation for maze-navigation runs.

``evaluate_outcome_fitness`` is the default for network-only evolution. It
scores only outcomes: progress along the maze's shortest path to the goal, a
bonus for reaching it, and a penalty for getting stuck. It does not reward or
penalize any particular driving style.

``evaluate_shaped_fitness`` is the older, behavior-shaped score (clearance,
exploration, collision, and backtracking terms). ``evaluate_fitness`` remains
as an alias to it for existing callers such as the MAP-Elites archive.
"""

from collections import OrderedDict, deque

import numpy as np

PATH_CELL_SIZE = 0.25
# Least-recently-used cache of distance maps; bounded because training on
# freshly generated mazes would otherwise grow it without limit.
PATH_MAP_CACHE_SIZE = 64
_PATH_MAP_CACHE = OrderedDict()


def _distance_to_goal(position, goal):
	position_x, position_y = position
	return ((position_x - goal[0])**2 + (position_y - goal[1])**2)**0.5


def _is_stuck(world, stuck_window, progress_tolerance):
	if world.reached_goal:
		return False
	if world.steps == 0 or world.moving_steps == 0:
		return True
	if world.steps < stuck_window:
		return False

	first_index = len(world.position_history) - stuck_window - 1
	starting_distance = _distance_to_goal(world.position_history[first_index], world.goal)
	best_distance = min(
		_distance_to_goal(position, world.goal)
		for position in world.position_history[first_index + 1:]
	)
	return starting_distance - best_distance < progress_tolerance


def _exploration_metrics(world, cell_size):
	cell_path = []
	for x_pos, y_pos in world.position_history:
		column = int((x_pos - world.maze.x_min) / cell_size)
		row = int((y_pos - world.maze.y_min) / cell_size)
		cell = (column, row)
		if not cell_path or cell != cell_path[-1]:
			cell_path.append(cell)

	visited_cells = set()
	path_revisits = 0
	for cell in cell_path:
		if cell in visited_cells:
			path_revisits += 1
		else:
			visited_cells.add(cell)

	total_cells = (
		int((world.maze.width + cell_size - 1e-12) / cell_size)
		* int((world.maze.height + cell_size - 1e-12) / cell_size)
	)
	coverage = min(1.0, len(visited_cells) / total_cells)
	return coverage, len(visited_cells), path_revisits


def _path_map_key(maze, radius, cell_size):
	return (
		maze.x_min, maze.y_min, maze.width, maze.height,
		tuple(maze.interior_walls), tuple(maze.obstacles), tuple(maze.goal),
		float(radius), float(cell_size),
	)


def _collision_grid(maze, columns, rows, cell_size, radius):
	"""Vectorized ``maze.collides`` for every grid-cell centre (same rules)."""
	xs = maze.x_min + (np.arange(columns) + 0.5) * cell_size
	ys = maze.y_min + (np.arange(rows) + 0.5) * cell_size
	x_pos, y_pos = np.meshgrid(xs, ys, indexing="ij")
	blocked = (
		(x_pos - radius < maze.x_min) | (x_pos + radius > maze.x_max)
		| (y_pos - radius < maze.y_min) | (y_pos + radius > maze.y_max)
	)
	radius_squared = radius**2
	for (start_x, start_y), (end_x, end_y) in maze.interior_walls:
		segment_x = end_x - start_x
		segment_y = end_y - start_y
		length_squared = segment_x**2 + segment_y**2
		if length_squared == 0:
			projection = np.zeros_like(x_pos)
		else:
			projection = np.clip(
				((x_pos - start_x) * segment_x + (y_pos - start_y) * segment_y)
				/ length_squared,
				0.0,
				1.0,
			)
		distance_squared = (
			(x_pos - (start_x + projection * segment_x))**2
			+ (y_pos - (start_y + projection * segment_y))**2
		)
		blocked |= distance_squared <= radius_squared
	for x_min, y_min, x_max, y_max in maze.obstacles:
		nearest_x = np.clip(x_pos, x_min, x_max)
		nearest_y = np.clip(y_pos, y_min, y_max)
		blocked |= (x_pos - nearest_x)**2 + (y_pos - nearest_y)**2 <= radius_squared
	return blocked


def path_distance_map(maze, radius, cell_size=PATH_CELL_SIZE):
	"""Return a grid of shortest-path distances (maze units) to the goal.

	The maze is divided into ``cell_size`` squares. A cell is free when a body
	of ``radius - cell_size`` centred there does not collide with a wall or
	obstacle; the one-cell allowance covers the robot sitting anywhere inside
	the cell, so gaps the robot can just squeeze through (for example exactly
	its own diameter) stay connected. A 4-connected breadth-first search from
	the goal cell gives the number of moves to the goal, multiplied by
	``cell_size``. Any cell the search does not reach (solid interiors) takes
	the distance of the nearest reached cell, so every lookup is finite.
	Results are cached per maze geometry.
	"""
	if cell_size <= 0:
		raise ValueError("cell_size must be positive")
	key = _path_map_key(maze, radius, cell_size)
	if key in _PATH_MAP_CACHE:
		_PATH_MAP_CACHE.move_to_end(key)
		return _PATH_MAP_CACHE[key]

	columns = max(1, int(round(maze.width / cell_size)))
	rows = max(1, int(round(maze.height / cell_size)))
	free = ~_collision_grid(maze, columns, rows, cell_size, max(0.0, radius - cell_size))

	steps = np.full((columns, rows), np.inf)
	goal_column = min(columns - 1, max(0, int((maze.goal[0] - maze.x_min) / cell_size)))
	goal_row = min(rows - 1, max(0, int((maze.goal[1] - maze.y_min) / cell_size)))
	steps[goal_column, goal_row] = 0.0
	queue = deque([(goal_column, goal_row)])
	while queue:
		column, row = queue.popleft()
		for d_column, d_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
			next_column, next_row = column + d_column, row + d_row
			if (
				0 <= next_column < columns and 0 <= next_row < rows
				and free[next_column, next_row]
				and steps[next_column, next_row] == np.inf
			):
				steps[next_column, next_row] = steps[column, row] + 1
				queue.append((next_column, next_row))

	# Give every unreached cell the distance of the nearest reached cell so a
	# body pressed into a wall (or any unexpected position) scores finitely.
	distances = steps * cell_size
	while not np.isfinite(distances).all():
		padded = np.pad(distances, 1, constant_values=np.inf)
		neighbours = np.minimum.reduce([
			padded[2:, 1:-1], padded[:-2, 1:-1], padded[1:-1, 2:], padded[1:-1, :-2],
		])
		distances = np.where(np.isfinite(distances), distances, neighbours)

	_PATH_MAP_CACHE[key] = distances
	if len(_PATH_MAP_CACHE) > PATH_MAP_CACHE_SIZE:
		_PATH_MAP_CACHE.popitem(last=False)
	return distances


def path_distances(maze, positions, radius, cell_size=PATH_CELL_SIZE):
	"""Look up shortest-path distances to the goal for ``(x, y)`` positions."""
	distances = path_distance_map(maze, radius, cell_size)
	points = np.asarray(positions, dtype=float).reshape(-1, 2)
	columns = np.clip(((points[:, 0] - maze.x_min) / cell_size).astype(int),
					  0, distances.shape[0] - 1)
	rows = np.clip(((points[:, 1] - maze.y_min) / cell_size).astype(int),
				   0, distances.shape[1] - 1)
	return distances[columns, rows]


def evaluate_outcome_fitness(
	world,
	*,
	goal_bonus=1.0,
	stuck_steps=300,
	stuck_penalty=0.2,
	cell_size=PATH_CELL_SIZE,
):
	"""Score a completed run on outcomes only.

	``score = path_progress + goal_bonus * reached_goal - stuck_penalty * stuck``

	``path_progress`` is the drop in shortest-path (BFS) distance to the goal
	from the start to the final position, divided by the starting distance;
	it is 1.0 when the goal is reached and negative if the robot ends farther
	along the path than it started. A run is ``stuck`` when it did not reach
	the goal and its best path distance has not improved during the last
	``stuck_steps`` steps.

	Collisions, steps, path length, and visited cells are reported as
	statistics only; they do not affect the score.
	"""
	if stuck_steps < 1:
		raise ValueError("stuck_steps must be at least 1")
	if goal_bonus < 0 or stuck_penalty < 0:
		raise ValueError("goal_bonus and stuck_penalty must be non-negative")

	radius = world.vehicle.radius
	distances = path_distances(world.maze, world.position_history, radius, cell_size)
	start_distance = float(distances[0])
	reached_goal = bool(world.reached_goal)
	final_distance = 0.0 if reached_goal else float(distances[-1])
	if not np.isfinite(start_distance):
		raise ValueError("the start position has no path to the goal")
	if start_distance == 0:
		path_progress = 1.0 if reached_goal else 0.0
		best_path_progress = path_progress
	else:
		path_progress = (start_distance - final_distance) / start_distance
		best_distance = 0.0 if reached_goal else float(np.min(distances))
		best_path_progress = (start_distance - best_distance) / start_distance

	running_best = np.minimum.accumulate(distances)
	improved = np.flatnonzero(np.diff(running_best) < 0)
	last_improvement = int(improved[-1]) + 1 if improved.size else 0
	steps_without_progress = len(distances) - 1 - last_improvement
	stuck = (not reached_goal) and steps_without_progress >= stuck_steps

	score_components = {
		"path_progress": path_progress,
		"goal_completion": goal_bonus if reached_goal else 0.0,
		"stuck": -stuck_penalty if stuck else 0.0,
	}
	coverage, unique_cells_visited, path_revisits = _exploration_metrics(world, 1.0)
	path_distance = sum(
		((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
		for start, end in zip(world.position_history, world.position_history[1:])
	)
	return {
		"score": sum(score_components.values()),
		"score_components": score_components,
		"reached_goal": reached_goal,
		"path_progress": path_progress,
		"best_path_progress": best_path_progress,
		"start_path_distance": start_distance,
		"final_path_distance": final_distance,
		"stuck": stuck,
		"steps_without_progress": steps_without_progress,
		# statistics only -- not scored
		"collisions": world.collisions,
		"steps": world.steps,
		"moving_steps": world.moving_steps,
		"path_distance": path_distance,
		"unique_cells_visited": unique_cells_visited,
		"coverage": coverage,
		"path_revisits": path_revisits,
	}


def evaluate_shaped_fitness(
	world,
	*,
	stuck_window=100,
	progress_tolerance=0.25,
	cell_size=1.0,
	progress_weight=2.0,
	goal_approach_weight=0.5,
	goal_regression_penalty=0.75,
	path_distance_weight=0.25,
	path_distance_cap=20.0,
	collision_penalty=2.0,
	repeated_collision_penalty=1.0,
	moving_step_penalty=0.001,
	stuck_penalty=20.0,
	goal_bonus=5.0,
	safe_distance=1.5,
	safe_distance_weight=4.0,
	exploration_weight=5.0,
	revisit_penalty=0.5,
):
	"""Score a completed ``World`` run and return metrics for analysis.

	The score rewards distance progress and reaching the goal, while charging
	for collisions and movement time. A run is stuck if it fails to make
	meaningful goal progress over ``stuck_window`` steps (or never moves in a
	shorter run).

	The ``behavior_descriptor`` contains normalized maze coverage and goal
	progress, suitable as the two axes of a MAP-Elites archive.
	"""
	if stuck_window < 1:
		raise ValueError("stuck_window must be at least 1")
	if progress_tolerance < 0:
		raise ValueError("progress_tolerance must be non-negative")
	if cell_size <= 0:
		raise ValueError("cell_size must be positive")
	if path_distance_cap <= 0:
		raise ValueError("path_distance_cap must be positive")
	if safe_distance <= 0:
		raise ValueError("safe_distance must be positive")
	if exploration_weight < 0 or revisit_penalty < 0:
		raise ValueError("exploration weights must be non-negative")

	final_position = world.position_history[-1]
	final_distance = _distance_to_goal(final_position, world.goal)
	progress = world.initial_distance - final_distance
	distances_to_goal = [
		_distance_to_goal(position, world.goal)
		for position in world.position_history
	]
	goal_approach_distance = sum(
		max(0.0, previous - current)
		for previous, current in zip(distances_to_goal, distances_to_goal[1:])
	)
	goal_regression_distance = sum(
		max(0.0, current - previous)
		for previous, current in zip(distances_to_goal, distances_to_goal[1:])
	)
	path_distance = sum(
		((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
		for start, end in zip(world.position_history, world.position_history[1:])
	)
	if world.initial_distance == 0:
		normalized_progress = 1.0 if world.reached_goal else 0.0
	else:
		normalized_progress = min(1.0, max(0.0, progress / world.initial_distance))

	reached_goal = final_distance <= world.goal_tolerance
	goal_bonus_awarded = goal_bonus if reached_goal else 0.0
	goal_overlap_margin = world.goal_tolerance - final_distance
	stuck = _is_stuck(world, stuck_window, progress_tolerance)
	coverage, unique_cells_visited, path_revisits = _exploration_metrics(world, cell_size)
	clearances = [
		world.maze.clearance(position, world.vehicle.radius)
		for position in world.position_history
	]
	safe_distance_score = sum(
		min(1.0, max(0.0, clearance) / safe_distance)
		for clearance in clearances
	) / len(clearances)
	repeated_collision_cost = repeated_collision_penalty * max(0, world.collisions - 1)**2

	score_components = {
		"goal_progress": progress_weight * progress,
		"goal_approach": goal_approach_weight * goal_approach_distance,
		"goal_regression": -goal_regression_penalty * goal_regression_distance,
		"travel": path_distance_weight * min(path_distance, path_distance_cap),
		"collision": -collision_penalty * world.collisions,
		"repeated_collision": -repeated_collision_cost,
		"movement_time": -moving_step_penalty * world.moving_steps,
		"stuck": -stuck_penalty * int(stuck),
		"safe_clearance": safe_distance_weight * safe_distance_score,
		"new_area": exploration_weight * coverage,
		"revisited_path": -revisit_penalty * path_revisits,
		"goal_completion": goal_bonus_awarded,
	}
	score = sum(score_components.values())
	return {
		"score": score,
		"score_components": score_components,
		"progress": progress,
		"goal_approach_distance": goal_approach_distance,
		"goal_regression_distance": goal_regression_distance,
		"path_distance": path_distance,
		"final_distance": final_distance,
		"goal_center_error": final_distance,
		"goal_tolerance": world.goal_tolerance,
		"goal_overlap_margin": goal_overlap_margin,
		"goal_bonus_awarded": goal_bonus_awarded,
		"collisions": world.collisions,
		"repeated_collision_cost": repeated_collision_cost,
		"moving_steps": world.moving_steps,
		"steps": world.steps,
		"stuck": stuck,
		"reached_goal": reached_goal,
		"minimum_clearance": min(clearances),
		"mean_clearance": sum(clearances) / len(clearances),
		"safe_distance_score": safe_distance_score,
		"coverage": coverage,
		"unique_cells_visited": unique_cells_visited,
		"path_revisits": path_revisits,
		"normalized_goal_progress": normalized_progress,
		"behavior_descriptor": (coverage, normalized_progress),
	}


# Backwards-compatible name for the shaped score (used by MAP-Elites).
evaluate_fitness = evaluate_shaped_fitness
