"""Fitness evaluation for maze-navigation runs."""


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


def _coverage(world, cell_size):
	visited_cells = set()
	for x_pos, y_pos in world.position_history:
		column = int((x_pos - world.maze.x_min) / cell_size)
		row = int((y_pos - world.maze.y_min) / cell_size)
		visited_cells.add((column, row))

	total_cells = (
		int((world.maze.width + cell_size - 1e-12) / cell_size)
		* int((world.maze.height + cell_size - 1e-12) / cell_size)
	)
	return min(1.0, len(visited_cells) / total_cells)


def evaluate_fitness(
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
	moving_step_penalty=0.01,
	stuck_penalty=20.0,
	goal_bonus=100.0,
	safe_distance=1.5,
	safe_distance_weight=4.0,
	wall_following_distance=2.5,
	preferred_wall_distance=1.0,
	wall_following_weight=2.0,
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
	if wall_following_distance <= 0:
		raise ValueError("wall_following_distance must be positive")
	if not 0 <= preferred_wall_distance <= wall_following_distance:
		raise ValueError("preferred_wall_distance must be within wall_following_distance")

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

	reached_goal = final_distance <= world.goal_radius
	stuck = _is_stuck(world, stuck_window, progress_tolerance)
	coverage = _coverage(world, cell_size)
	clearances = [
		world.maze.clearance(position, world.vehicle.radius)
		for position in world.position_history
	]
	safe_distance_score = sum(
		min(1.0, max(0.0, clearance) / safe_distance)
		for clearance in clearances
	) / len(clearances)
	wall_scores = [
		world.maze.wall_following_score(
			start,
			end,
			radius=world.vehicle.radius,
			follow_distance=wall_following_distance,
			preferred_distance=preferred_wall_distance,
		)
		for start, end in zip(world.position_history, world.position_history[1:])
	]
	wall_following_score = sum(wall_scores) / len(wall_scores) if wall_scores else 0.0
	repeated_collision_cost = repeated_collision_penalty * max(0, world.collisions - 1)**2

	score = (
		progress_weight * progress
		+ goal_approach_weight * goal_approach_distance
		- goal_regression_penalty * goal_regression_distance
		+ path_distance_weight * min(path_distance, path_distance_cap)
		- collision_penalty * world.collisions
		- repeated_collision_cost
		- moving_step_penalty * world.moving_steps
		- stuck_penalty * int(stuck)
		+ safe_distance_weight * safe_distance_score
		+ wall_following_weight * wall_following_score
		+ goal_bonus * int(reached_goal)
	)
	return {
		"score": score,
		"progress": progress,
		"goal_approach_distance": goal_approach_distance,
		"goal_regression_distance": goal_regression_distance,
		"path_distance": path_distance,
		"final_distance": final_distance,
		"collisions": world.collisions,
		"repeated_collision_cost": repeated_collision_cost,
		"moving_steps": world.moving_steps,
		"steps": world.steps,
		"stuck": stuck,
		"reached_goal": reached_goal,
		"minimum_clearance": min(clearances),
		"mean_clearance": sum(clearances) / len(clearances),
		"safe_distance_score": safe_distance_score,
		"wall_following_score": wall_following_score,
		"coverage": coverage,
		"normalized_goal_progress": normalized_progress,
		"behavior_descriptor": (coverage, normalized_progress),
	}
