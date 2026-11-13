"""Hand-written wall followers on the simple setup: 50 random 12x12 test mazes + courses A/B/C.

Same conditions as the evolved controllers' test: physical motor convention,
distance sensors at -60/0/+60, default robot noise + heading noise 0.05, 8x
limit. Each run goes to CAP steps; "within limit" is read from the same run.
Usage: python wf_simple.py REPO OUT_DIR
"""

import csv
import sys
from multiprocessing import Pool

REPO, OUT = sys.argv[1:3]
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.experiment import course_step_limit, make_evaluation_courses, sample_random_mazes  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402
from wall_follower import WallFollower  # noqa: E402

CAP = 20000
TEST_SEED = 2026
N_TEST = 50
COURSE_TRIALS = 10


def entries():
    out = []
    for i, rm in enumerate(sample_random_mazes(np.random.default_rng(TEST_SEED), N_TEST)):
        out.append(("random_12x12", f"r{i}", rm.maze, rm.start, rm.start_heading, 700000 + i))
    for name, _, maze, start, heading in make_evaluation_courses():
        for t in range(COURSE_TRIALS):
            out.append((name, f"t{t}", maze, start, heading, 500000 + t))
    return out


def run(job):
    set_name, label, maze, start, heading, noise_seed, side = job
    vehicle = Vehicle(controller=WallFollower(side), noise_stdev=0.05, noise_seed=noise_seed,
                      navigation_heuristics=False, start_position=start, start_heading=heading,
                      **DEFAULT_ROBOT_NOISE)
    world = World(maze=maze, vehicle=vehicle)
    limit = course_step_limit(maze, start=start)
    world.run(CAP)
    goal = bool(world.reached_goal)
    return {"controller": f"{side}_wall", "set": set_name, "maze": label, "limit": limit,
            "goal_within_limit": int(goal and world.steps <= limit), "goal_no_limit": int(goal),
            "steps": world.steps, "steps_over_limit": round(world.steps / limit, 2) if goal else "",
            "pct_steps_in_collision": round(100 * world.collisions / max(world.steps, 1), 1)}


if __name__ == "__main__":
    jobs = [e + (side,) for e in entries() for side in ("left", "right")]
    with Pool(10) as pool:
        rows = pool.map(run, jobs)
    with open(f"{OUT}/wall_followers_simple.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for set_name in ("random_12x12", "course_A", "course_B", "course_C"):
        for c in ("left_wall", "right_wall"):
            sel = [r for r in rows if r["set"] == set_name and r["controller"] == c]
            ratios = [r["steps_over_limit"] for r in sel if r["goal_no_limit"]]
            print(f"{set_name:13} {c:10} within 8x limit {100 * np.mean([r['goal_within_limit'] for r in sel]):5.1f}%  "
                  f"no limit {100 * np.mean([r['goal_no_limit'] for r in sel]):5.1f}%  "
                  f"median limit {int(np.median([r['limit'] for r in sel]))}  "
                  f"median steps/limit {np.median(ratios) if ratios else float('nan'):.2f}  "
                  f"coll {np.mean([r['pct_steps_in_collision'] for r in sel]):.0f}%")
