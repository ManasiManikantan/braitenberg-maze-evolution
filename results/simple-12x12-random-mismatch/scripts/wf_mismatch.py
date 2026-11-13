"""Does the fixed motor mismatch cause the left/right wall-follower gap?

Runs both hand-written wall followers on the simple-setup test set (50 random
12x12 mazes + courses A/B/C x 10) under the 8x limit with motor_mismatch set
to +0.05 (default: left wheel stronger), 0, and -0.05 (right wheel stronger).
Extra keyword overrides for Vehicle can be given as JSON (argv[3]).
Usage: python wf_mismatch.py REPO OUT_DIR [overrides_json]
"""

import json
import sys
from multiprocessing import Pool

REPO, OUT = sys.argv[1:3]
OVERRIDES = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.experiment import course_step_limit, make_evaluation_courses, sample_random_mazes  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
from wall_follower import WallFollower  # noqa: E402


def entries():
    out = [("random_12x12", rm.maze, rm.start, rm.start_heading, 700000 + i)
           for i, rm in enumerate(sample_random_mazes(np.random.default_rng(2026), 50))]
    for name, _, maze, start, heading in make_evaluation_courses():
        out += [(name, maze, start, heading, 500000 + t) for t in range(10)]
    return out


def run(job):
    set_name, maze, start, heading, noise_seed, side, overrides = job
    noise = {**DEFAULT_ROBOT_NOISE, **overrides}
    vehicle = Vehicle(controller=WallFollower(side), noise_stdev=0.05, noise_seed=noise_seed,
                      navigation_heuristics=False, start_position=start, start_heading=heading, **noise)
    world = World(maze=maze, vehicle=vehicle)
    world.run(course_step_limit(maze, start=start))
    return set_name, side, json.dumps(overrides), int(world.reached_goal)


if __name__ == "__main__":
    variants = [OVERRIDES] if OVERRIDES is not None else [
        {"motor_mismatch": 0.05}, {"motor_mismatch": 0.0}, {"motor_mismatch": -0.05}]
    jobs = [e + (side, v) for v in variants for e in entries() for side in ("left", "right")]
    with Pool(10) as pool:
        results = pool.map(run, jobs)
    for v in variants:
        key = json.dumps(v)
        line = []
        for set_name in ("random_12x12", "course_A", "course_B", "course_C"):
            for side in ("left", "right"):
                sel = [g for s, sd, k, g in results if s == set_name and sd == side and k == key]
                line.append(f"{set_name[-6:] if set_name.startswith('course') else 'random'} {side[0].upper()} "
                            f"{100 * np.mean(sel):3.0f}%")
        print(f"{key:40} " + " | ".join(line))
