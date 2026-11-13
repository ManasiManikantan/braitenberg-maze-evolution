"""Behavior and performance measures of recorded runs (e.g. for MAP-Elites).

``record_run`` steps one controller through one maze and keeps what each step
looked like; ``run_metrics`` turns that record into measures; and
``measure_controller`` repeats this over many mazes and starts and returns
the mean and standard deviation of every measure.

Behavior measures (all dimensionless):

- ``speed``: mean forward speed / maximum forward speed, in [0, 1]. Reversing
  counts as 0; a blocked (colliding) step has speed 0.
- ``wall_clearance``: mean over steps of the closest distance-sensor reading /
  sensor range, in [0, 1] (0 = touching something ahead or to the side).
- ``turning_bias``: sum of turn rates / sum of |turn rates|, in [-1, 1].
  Positive = turns left (counter-clockwise), the simulator's convention.
- ``exploration``: maze cells visited / cells reachable from the start, in
  [0, 1], on a grid of ``cell_size`` squares (default 4, one corridor cell of
  the generated mazes).
- ``wall_side``: of the steps where a wall is close (within ``close_distance``,
  default 1 unit, of the body measured straight out to the side) on exactly
  one side, the fraction where it is on the left, in [0, 1]. About 0.8 for
  the hand-written left-wall follower and 0.2 for the right-wall follower;
  NaN if no step had a wall on exactly one side.

Performance measures: ``goal_reached`` (0/1), ``path_progress`` (drop in
shortest-path distance to the goal / starting distance, as in the
outcome-only fitness), ``steps``, and ``pct_steps_in_collision``.
"""

import math
from collections import deque

import numpy as np

from evolution.experiment import DEFAULT_NOISE_STDEV, course_step_limit, sample_random_mazes
from evolution.fitness import PATH_CELL_SIZE, _collision_grid, path_distances
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle
from sim.world import World

BEHAVIOR_MEASURES = ("speed", "wall_clearance", "turning_bias", "exploration", "wall_side")
PERFORMANCE_MEASURES = ("goal_reached", "path_progress", "steps", "pct_steps_in_collision")
EXPLORATION_CELL_SIZE = 4.0
WALL_SIDE_CLOSE_DISTANCE = 1.0


class RunRecord:
    """Per-step record of one run."""

    def __init__(self, maze, positions, headings, speeds, collided, sensor_distances,
                 reached_goal, sensor_range, max_speed, radius, side_clearances=None):
        self.maze = maze
        self.positions = np.asarray(positions, dtype=float)        # (steps + 1, 2)
        self.headings = np.asarray(headings, dtype=float)          # (steps + 1,)
        self.speeds = np.asarray(speeds, dtype=float)              # (steps,)
        self.collided = np.asarray(collided, dtype=bool)           # (steps,)
        self.sensor_distances = np.asarray(sensor_distances, dtype=float).reshape(-1, 3)
        self.reached_goal = bool(reached_goal)
        self.sensor_range = float(sensor_range)
        self.max_speed = float(max_speed)
        self.radius = float(radius)
        # (steps, 2): wall clearance straight out to the left and right of the body
        self.side_clearances = (np.full((len(self.speeds), 2), np.inf) if side_clearances is None
                                else np.asarray(side_clearances, dtype=float).reshape(-1, 2))

    @property
    def steps(self):
        return len(self.speeds)


def record_run(controller, maze, start=(0.0, 0.0), heading=0.0, max_steps=None, noise_seed=None,
               noise_stdev=DEFAULT_NOISE_STDEV, robot_noise=None, light_occlusion="blocked",
               navigation_heuristics=False):
    """Run ``controller`` in ``maze`` and return a ``RunRecord``.

    ``max_steps`` defaults to the maze's ``course_step_limit`` from ``start``.
    ``light_occlusion`` should match the controller's inputs (``None`` for a
    3-input controller; controllers that read only the first three inputs,
    like the wall followers, work with either).
    """
    robot_noise = DEFAULT_ROBOT_NOISE if robot_noise is None else robot_noise
    vehicle = Vehicle(controller=controller, noise_stdev=noise_stdev, noise_seed=noise_seed,
                      navigation_heuristics=navigation_heuristics, start_position=start,
                      start_heading=heading, light_occlusion=light_occlusion, **robot_noise)
    world = World(maze=maze, vehicle=vehicle)
    if max_steps is None:
        max_steps = course_step_limit(maze, start=start)
    headings = [vehicle.orientation]
    speeds, collided, sensors, sides = [], [], [], []
    while world.steps < max_steps and not world.reached_goal:
        before = world.collisions
        world.step()
        headings.append(vehicle.orientation)
        speeds.append(vehicle.velocity)
        collided.append(world.collisions > before)
        sensors.append((vehicle.left_distance, vehicle.front_distance, vehicle.right_distance))
        h = vehicle.orientation
        centre = (vehicle.x_pos, vehicle.y_pos)
        sides.append((
            maze.raycast(centre, (-math.sin(h), math.cos(h)), vehicle.sensor_range) - vehicle.radius,
            maze.raycast(centre, (math.sin(h), -math.cos(h)), vehicle.sensor_range) - vehicle.radius,
        ))
    return RunRecord(maze, world.position_history, headings, speeds, collided, sensors,
                     world.reached_goal, vehicle.sensor_range, vehicle.vel_gain, vehicle.radius,
                     side_clearances=sides)


def _reachable_cells(maze, start, radius, cell_size):
    """Coarse ``cell_size`` cells containing a fine grid point reachable from ``start``."""
    fine = PATH_CELL_SIZE
    columns = max(1, int(round(maze.width / fine)))
    rows = max(1, int(round(maze.height / fine)))
    free = ~_collision_grid(maze, columns, rows, fine, max(0.0, radius - fine))
    i = min(columns - 1, max(0, int((start[0] - maze.x_min) / fine)))
    j = min(rows - 1, max(0, int((start[1] - maze.y_min) / fine)))
    if not free[i, j]:
        # start touching a wall: begin from the nearest free fine cell
        candidates = np.argwhere(free)
        i, j = candidates[np.argmin(np.hypot(candidates[:, 0] - i, candidates[:, 1] - j))]
    seen = np.zeros_like(free)
    seen[i, j] = True
    queue = deque([(i, j)])
    while queue:
        a, b = queue.popleft()
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            na, nb = a + da, b + db
            if 0 <= na < columns and 0 <= nb < rows and free[na, nb] and not seen[na, nb]:
                seen[na, nb] = True
                queue.append((na, nb))
    points = np.argwhere(seen)
    xs = maze.x_min + (points[:, 0] + 0.5) * fine
    ys = maze.y_min + (points[:, 1] + 0.5) * fine
    return {_cell((x, y), maze, cell_size) for x, y in zip(xs, ys)}


def _cell(position, maze, cell_size):
    return (int((position[0] - maze.x_min) // cell_size), int((position[1] - maze.y_min) // cell_size))


def run_metrics(record, cell_size=EXPLORATION_CELL_SIZE, close_distance=WALL_SIDE_CLOSE_DISTANCE):
    """Return the behavior and performance measures of one ``RunRecord``."""
    steps = record.steps
    if steps == 0:
        speed = clearance = bias = 0.0
    else:
        speed = float(np.clip(np.mean(np.maximum(record.speeds, 0.0)) / record.max_speed, 0.0, 1.0))
        clearance = float(np.mean(record.sensor_distances.min(axis=1)) / record.sensor_range)
        turns = np.diff(record.headings)
        turns = (turns + np.pi) % (2 * np.pi) - np.pi
        total = float(np.sum(np.abs(turns)))
        bias = float(np.sum(turns) / total) if total > 0 else 0.0

    left_close = record.side_clearances[:, 0] < close_distance
    right_close = record.side_clearances[:, 1] < close_distance
    one_sided = left_close ^ right_close
    wall_side = float(np.mean(left_close[one_sided])) if one_sided.any() else float("nan")

    reachable = _reachable_cells(record.maze, tuple(record.positions[0]), record.radius, cell_size)
    visited = {_cell(p, record.maze, cell_size) for p in record.positions}
    exploration = len(visited & reachable) / max(len(reachable), 1)

    distances = path_distances(record.maze, record.positions[[0, -1]], record.radius)
    start_distance = float(distances[0])
    final_distance = 0.0 if record.reached_goal else float(distances[1])
    if start_distance > 0:
        progress = (start_distance - final_distance) / start_distance
    else:
        progress = 1.0 if record.reached_goal else 0.0
    return {
        "speed": speed,
        "wall_clearance": clearance,
        "turning_bias": bias,
        "exploration": float(exploration),
        "wall_side": wall_side,
        "goal_reached": float(record.reached_goal),
        "path_progress": float(progress),
        "steps": float(steps),
        "pct_steps_in_collision": 100.0 * float(np.mean(record.collided)) if steps else 0.0,
    }


def _nan_stat(function, values):
    values = np.asarray(values, dtype=float)
    return float(function(values)) if np.isfinite(values).any() else float("nan")


def random_maze_episodes(count, seed, maze_options=None, noise_seed_base=0):
    """``count`` episodes ``(maze, start, heading, noise_seed)`` on random mazes."""
    mazes = sample_random_mazes(np.random.default_rng(seed), count, maze_options=maze_options)
    return [(rm.maze, rm.start, rm.start_heading, noise_seed_base + index)
            for index, rm in enumerate(mazes)]


def measure_controller(controller, episodes, cell_size=EXPLORATION_CELL_SIZE, **run_options):
    """Measure ``controller`` over ``episodes`` and summarize every measure.

    ``episodes`` is a list of ``(maze, start, heading, noise_seed)`` (see
    ``random_maze_episodes``); ``run_options`` are passed to ``record_run``.
    Returns ``{"mean": {...}, "std": {...}, "runs": [per-run dicts]}``; runs
    where a measure is NaN (``wall_side`` with no one-sided wall) are skipped
    for that measure.
    """
    runs = [
        run_metrics(record_run(controller, maze, start, heading, noise_seed=noise_seed,
                               **run_options), cell_size)
        for maze, start, heading, noise_seed in episodes
    ]
    names = BEHAVIOR_MEASURES + PERFORMANCE_MEASURES
    return {
        "mean": {name: _nan_stat(np.nanmean, [run[name] for run in runs]) for name in names},
        "std": {name: _nan_stat(np.nanstd, [run[name] for run in runs]) for name in names},
        "runs": runs,
    }


__all__ = [
    "BEHAVIOR_MEASURES", "PERFORMANCE_MEASURES", "RunRecord", "measure_controller",
    "random_maze_episodes", "record_run", "run_metrics",
]
