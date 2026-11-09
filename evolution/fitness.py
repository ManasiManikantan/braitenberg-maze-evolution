"""Fitness evaluation for maze-navigation runs."""


def _distance_to_goal(position, goal):
	position_x, position_y = position
	return ((position_x - goal[0])**2 + (position_y - goal[1])**2)**0.5


def _is_stuck(world, stuck_window, progress_tolerance):
	if world.reached_goal or world.steps == 0:
		return False
	if world.steps < stuck_window:
		return world.moving_steps == 0

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
	progress_weight=1.0,
	collision_penalty=2.0,
	moving_step_penalty=0.01,
	stuck_penalty=5.0,
	goal_bonus=10.0,
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

	final_position = world.position_history[-1]
	final_distance = _distance_to_goal(final_position, world.goal)
	progress = world.initial_distance - final_distance
	if world.initial_distance == 0:
		normalized_progress = 1.0 if world.reached_goal else 0.0
	else:
		normalized_progress = min(1.0, max(0.0, progress / world.initial_distance))

	reached_goal = final_distance <= world.goal_radius
	stuck = _is_stuck(world, stuck_window, progress_tolerance)
	coverage = _coverage(world, cell_size)

	score = (
		progress_weight * progress
		- collision_penalty * world.collisions
		- moving_step_penalty * world.moving_steps
		- stuck_penalty * int(stuck)
		+ goal_bonus * int(reached_goal)
	)
	return {
		"score": score,
		"progress": progress,
		"final_distance": final_distance,
		"collisions": world.collisions,
		"moving_steps": world.moving_steps,
		"steps": world.steps,
		"stuck": stuck,
		"reached_goal": reached_goal,
		"coverage": coverage,
		"normalized_goal_progress": normalized_progress,
		"behavior_descriptor": (coverage, normalized_progress),
	}
