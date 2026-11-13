"""Diagnostic: hand-written left/right wall followers on random mazes and courses 1-4.

No evolution. Distance sensors only (no light), network-only physics, default
robot noise + heading noise 0.05. Each run goes up to CAP steps; condition (a)
"current limits" is read off the same run (did it reach the goal within the
maze's 2 x shortest-route limit?) since the trajectory prefix is identical.

Usage: python wall_follower_diag.py REPO OUT_DIR
"""

import csv
import math
import sys
from multiprocessing import Pool

REPO, OUT = sys.argv[1:3]
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.experiment import DEFAULT_CURRICULUM, course_step_limit  # noqa: E402
from evolution.fitness import path_distance_map, path_distances  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.maze_generator import generate_perfect_maze  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
from wall_follower import WallFollower  # noqa: E402

CAP = 20000
TEST_SEED = 2026
N_MAZES = 50
COURSE_TRIALS = 5
NARROW_WIDTH = 2.6   # passage width below which a 2-unit robot is in a "narrow gap"


def test_sets():
    sets = {}
    for _, options in DEFAULT_CURRICULUM:
        rng = np.random.default_rng(TEST_SEED)
        size = int(options["size"])
        sets[f"random_{size}x{size}"] = [
            (f"r{i}", rm.maze, rm.start, rm.start_heading, rm.route_cells)
            for i, rm in enumerate(generate_perfect_maze(rng, **options) for _ in range(N_MAZES))
        ]
    for d in (1, 2, 3, 4):
        maze = Maze(difficulty=d)
        sets[f"course_{d}"] = [(f"t{t}", maze, (0.0, 0.0), 0.0, None) for t in range(COURSE_TRIALS)]
    return sets


def passage_width(maze, position):
    """Narrowest wall-to-wall width through ``position`` over 36 directions."""
    widths = []
    for k in range(18):
        a = math.pi * k / 18
        direction = (math.cos(a), math.sin(a))
        opposite = (-direction[0], -direction[1])
        widths.append(maze.raycast(position, direction, 20.0) + maze.raycast(position, opposite, 20.0))
    return min(widths)


def run(job):
    set_name, label, maze, start, heading, route_cells, side, noise_seed = job
    vehicle = Vehicle(controller=WallFollower(side), noise_stdev=0.05, noise_seed=noise_seed,
                      navigation_heuristics=False, start_position=start, start_heading=heading,
                      **DEFAULT_ROBOT_NOISE)
    world = World(maze=maze, vehicle=vehicle)
    limit = course_step_limit(maze, start=start)
    world.run(CAP)
    path = np.array(world.position_history)
    goal = bool(world.reached_goal)
    row = {
        "controller": f"{side}_wall", "set": set_name, "maze": label, "route_cells": route_cells,
        "limit": limit, "goal_no_limit": int(goal), "steps": world.steps,
        "goal_within_limit": int(goal and world.steps <= limit),
        "steps_over_limit": round(world.steps / limit, 2) if goal else "",
        "pct_steps_in_collision": round(100 * world.collisions / max(world.steps, 1), 1),
        "failure_no_limit": "", "failure_with_limit": "", "end_x": round(path[-1, 0], 1),
        "end_y": round(path[-1, 1], 1), "end_passage_width": "",
    }
    if goal and world.steps > limit:
        row["failure_with_limit"] = "time-out (goal reached later)"
    if not goal:
        tail = path[-2000:]
        moved = float(np.linalg.norm(tail[-1] - tail[0])) if len(tail) > 1 else 0.0
        cells_before = {(int(x), int(y)) for x, y in path[:-5000]}
        new_late = {(int(x), int(y)) for x, y in path[-5000:]} - cells_before
        width = passage_width(maze, tuple(path[-1]))
        if moved < 1.0:
            kind = "stuck"
        elif not new_late:
            kind = "loop"
        else:
            kind = "still exploring at cap"
        if kind != "still exploring at cap" and width < NARROW_WIDTH:
            kind += " at narrow gap"
        row["failure_no_limit"] = kind
        row["failure_with_limit"] = kind
        row["end_passage_width"] = round(width, 2)
    return row


def shortest_route_profile(maze, start):
    """Route length (units) and narrowest wall-to-wall passage along the shortest route."""
    distances = path_distance_map(maze, 1.0)
    cell = 0.25
    i = int((start[0] - maze.x_min) / cell)
    j = int((start[1] - maze.y_min) / cell)
    widths = []
    for _ in range(4000):
        x, y = maze.x_min + (i + 0.5) * cell, maze.y_min + (j + 0.5) * cell
        widths.append(passage_width(maze, (x, y)))
        if distances[i, j] == 0:
            break
        neighbours = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= i + di < distances.shape[0] and 0 <= j + dj < distances.shape[1]]
        i, j = min(neighbours, key=lambda c: distances[c])
    return float(path_distances(maze, [start], 1.0)[0]), min(widths)


if __name__ == "__main__":
    sets = test_sets()
    jobs = []
    for set_name, entries in sets.items():
        for index, (label, maze, start, heading, route_cells) in enumerate(entries):
            for side in ("left", "right"):
                jobs.append((set_name, label, maze, start, heading, route_cells, side, 31000 + index))
    with Pool(10) as pool:
        rows = pool.map(run, jobs)
    with open(f"{OUT}/wall_follower_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("RESULTS (per controller and set)")
    summary = []
    for set_name in sets:
        for side in ("left_wall", "right_wall"):
            sel = [r for r in rows if r["set"] == set_name and r["controller"] == side]
            n = len(sel)
            goal_steps = [r["steps"] for r in sel if r["goal_no_limit"]]
            ratios = [r["steps_over_limit"] for r in sel if r["goal_no_limit"]]
            fail_a = {}
            fail_b = {}
            for r in sel:
                if not r["goal_within_limit"]:
                    fail_a[r["failure_with_limit"]] = fail_a.get(r["failure_with_limit"], 0) + 1
                if not r["goal_no_limit"]:
                    fail_b[r["failure_no_limit"]] = fail_b.get(r["failure_no_limit"], 0) + 1
            entry = {
                "set": set_name, "controller": side, "runs": n,
                "goal_within_current_limit": round(sum(r["goal_within_limit"] for r in sel) / n, 3),
                "goal_no_limit": round(sum(r["goal_no_limit"] for r in sel) / n, 3),
                "median_limit": int(np.median([r["limit"] for r in sel])),
                "median_steps_to_goal": int(np.median(goal_steps)) if goal_steps else "",
                "median_steps_over_limit": round(float(np.median(ratios)), 2) if ratios else "",
                "p90_steps_over_limit": round(float(np.percentile(ratios, 90)), 2) if ratios else "",
                "pct_steps_in_collision": round(float(np.mean([r["pct_steps_in_collision"] for r in sel])), 1),
                "failures_with_limit": "; ".join(f"{k}: {v}" for k, v in sorted(fail_a.items())),
                "failures_no_limit": "; ".join(f"{k}: {v}" for k, v in sorted(fail_b.items())),
            }
            summary.append(entry)
            print("  " + " | ".join(f"{k}={v}" for k, v in entry.items()))
    with open(f"{OUT}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print("\nSIZE COMPARISON (shortest robot route in units and in 4-unit cells; narrowest passage on it)")
    for set_name, entries in sets.items():
        profiles = [shortest_route_profile(maze, start) for _, maze, start, _, _ in entries[:50]]
        lengths = [p[0] for p in profiles]
        widths = [p[1] for p in profiles]
        maze = entries[0][1]
        print(f"  {set_name:16} area {maze.width:.0f}x{maze.height:.0f} ({maze.width / 4:.0f}x{maze.height / 4:.0f} cells): "
              f"route {np.median(lengths):.1f} units median ({min(lengths):.1f}-{max(lengths):.1f}) = "
              f"{np.median(lengths) / 4:.1f} cells; narrowest passage on route {min(widths):.2f} "
              f"(median {np.median(widths):.2f})")
