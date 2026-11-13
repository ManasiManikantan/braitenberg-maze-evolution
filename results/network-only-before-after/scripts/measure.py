"""Shared, fixed measurement used for both the before and after runs.

Path progress uses its own BFS distance map (0.25 grid, robot-radius
clearance, 4-connected) so the before/after numbers are measured identically
regardless of what the repo's fitness code does.
"""

import csv
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

CELL = 0.25
_CACHE = {}


def bfs_map(maze, radius=1.0):
    key = (maze.width, maze.height, tuple(maze.interior_walls), tuple(maze.obstacles),
           tuple(maze.goal), radius)
    if key in _CACHE:
        return _CACHE[key]
    nx = int(round(maze.width / CELL))
    ny = int(round(maze.height / CELL))
    free = np.zeros((nx, ny), bool)
    for i in range(nx):
        for j in range(ny):
            free[i, j] = not maze.collides(
                (maze.x_min + (i + .5) * CELL, maze.y_min + (j + .5) * CELL), radius - CELL)
    dist = np.full((nx, ny), np.inf)
    gi = int((maze.goal[0] - maze.x_min) / CELL)
    gj = int((maze.goal[1] - maze.y_min) / CELL)
    dist[gi, gj] = 0
    q = deque([(gi, gj)])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b] and dist[a, b] == np.inf:
                dist[a, b] = dist[i, j] + 1
                q.append((a, b))
    dist = dist * CELL
    while not np.isfinite(dist).all():
        padded = np.pad(dist, 1, constant_values=np.inf)
        neighbours = np.minimum.reduce([
            padded[2:, 1:-1], padded[:-2, 1:-1], padded[1:-1, 2:], padded[1:-1, :-2]])
        dist = np.where(np.isfinite(dist), dist, neighbours)
    _CACHE[key] = (dist, maze)
    return _CACHE[key]


def path_distance(maze, pos):
    dist, _ = bfs_map(maze)
    nx, ny = dist.shape
    i = min(nx - 1, max(0, int((pos[0] - maze.x_min) / CELL)))
    j = min(ny - 1, max(0, int((pos[1] - maze.y_min) / CELL)))
    return float(dist[i, j])


def measure_world(world):
    """Outcome stats for one finished World."""
    maze = world.maze
    start = path_distance(maze, world.position_history[0])
    end = 0.0 if world.reached_goal else path_distance(maze, world.position_history[-1])
    best = 0.0 if world.reached_goal else min(path_distance(maze, p) for p in world.position_history[::5])
    v = world.vehicle
    # spin detection: total heading change vs distance travelled
    path_len = sum(math.dist(a, b) for a, b in zip(world.position_history, world.position_history[1:]))
    return {
        "goal_success": int(world.reached_goal),
        "path_progress": (start - end) / start,
        "best_path_progress": (start - best) / start,
        "collisions": world.collisions,
        "steps": world.steps,
        "path_length": path_len,
    }


def summarize(rows):
    out = {}
    for course in sorted({r["course"] for r in rows}):
        rs = [r for r in rows if r["course"] == course]
        out[course] = {
            "n": len(rs),
            "goal_success": sum(r["goal_success"] for r in rs) / len(rs),
            "path_progress": float(np.mean([r["path_progress"] for r in rs])),
            "best_path_progress": float(np.mean([r["best_path_progress"] for r in rs])),
            "collisions": float(np.mean([r["collisions"] for r in rs])),
            "path_length": float(np.mean([r["path_length"] for r in rs])),
        }
    rs = rows
    out["all"] = {
        "n": len(rs),
        "goal_success": sum(r["goal_success"] for r in rs) / len(rs),
        "path_progress": float(np.mean([r["path_progress"] for r in rs])),
        "best_path_progress": float(np.mean([r["best_path_progress"] for r in rs])),
        "collisions": float(np.mean([r["collisions"] for r in rs])),
        "path_length": float(np.mean([r["path_length"] for r in rs])),
    }
    return out


def write(rows, history, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}_episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out_dir / f"{name}_history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    summary = summarize(rows)
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
