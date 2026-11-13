"""Check / grid-search wall-follower parameters on the 20 validation mazes (seed 777, 20x20).

Usage: python tune_wall_follower.py REPO [grid]
"""

import itertools
import sys
from multiprocessing import Pool

REPO = sys.argv[1]
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from sim.maze_generator import generate_perfect_maze  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
from wall_follower import WallFollower  # noqa: E402

rng = np.random.default_rng(777)
MAZES = [generate_perfect_maze(rng) for _ in range(20)]
CAP = 20000


def score(params):
    result = []
    for side in ("left", "right"):
        goals, collisions, steps, goal_steps = 0, 0, 0, []
        for i, rm in enumerate(MAZES):
            vehicle = Vehicle(controller=WallFollower(side, **params), noise_stdev=0.05, noise_seed=i,
                              navigation_heuristics=False, start_position=rm.start,
                              start_heading=rm.start_heading, **DEFAULT_ROBOT_NOISE)
            world = World(maze=rm.maze, vehicle=vehicle)
            world.run(CAP)
            goals += world.reached_goal
            collisions += world.collisions
            steps += world.steps
            if world.reached_goal:
                goal_steps.append(world.steps)
        result.append((goals, round(100 * collisions / steps), int(np.median(goal_steps)) if goal_steps else None))
    return params, result


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "grid":
        grid = [dict(near=n, far=f, arc_radius=r, front_stop=s)
                for n, f, r, s in itertools.product([0.6, 0.8], [1.8, 2.4], [1.6, 2.0], [1.4, 2.0])]
    else:
        grid = [{}]
    with Pool(10) as pool:
        out = pool.map(score, grid)
    for params, result in sorted(out, key=lambda item: -sum(g for g, _, _ in item[1])):
        print(params, "left (goals/20, coll%, median goal steps):", result[0], "right:", result[1])
