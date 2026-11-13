"""Mirror test: mirrored right-turners on course A, mirrored left-turners on course B.

Controllers: the 5 seeds of the random-mismatch rerun (60 generations).
Course trials use the same noise seeds as the earlier tests (500000 + t).
Turning bias of originals and mirrors is measured on the 100-maze metric set.
Usage: python mirror_test.py REPO GENOMES_NPZ OUT_CSV TRIALS
"""

import csv
import sys
from multiprocessing import Pool

REPO, GENOMES, OUT_CSV = sys.argv[1:4]
TRIALS = int(sys.argv[4])
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from controller.mirror import MirroredController  # noqa: E402
from evolution.experiment import course_step_limit, make_evaluation_courses  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from evolution.metrics import measure_controller, random_maze_episodes  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402

COURSES = {name: (maze, start, heading) for name, _, maze, start, heading in make_evaluation_courses()}


def controller_for(seed, mirrored):
    net = GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked").decode_genome(
        np.load(GENOMES)[f"seed{seed}"])
    return MirroredController(net) if mirrored else net


def course_job(args):
    seed, mirrored, course = args
    controller = controller_for(seed, mirrored)
    maze, start, heading = COURSES[course]
    goals, steps = 0, []
    for t in range(TRIALS):
        vehicle = Vehicle(controller=controller, noise_stdev=0.05, noise_seed=500000 + t,
                          navigation_heuristics=False, start_position=start, start_heading=heading,
                          light_occlusion="blocked", **DEFAULT_ROBOT_NOISE)
        world = World(maze=maze, vehicle=vehicle)
        world.run(course_step_limit(maze, start=start))
        goals += world.reached_goal
        if world.reached_goal:
            steps.append(world.steps)
    return seed, mirrored, course, goals, (int(np.median(steps)) if steps else "")


def bias_job(args):
    seed, mirrored = args
    summary = measure_controller(controller_for(seed, mirrored),
                                 random_maze_episodes(100, seed=4242, noise_seed_base=900000))
    return seed, mirrored, summary["mean"]["turning_bias"], summary["mean"]["goal_reached"]


if __name__ == "__main__":
    seeds = (1, 2, 3, 4)
    course_jobs = [(s, m, c) for s in seeds for m in (False, True) for c in ("course_A", "course_B")]
    bias_jobs = [(s, m) for s in seeds for m in (False, True)]
    with Pool(8) as pool:
        course_results = pool.map(course_job, course_jobs)
        bias_results = pool.map(bias_job, bias_jobs)
    bias = {(s, m): (b, g) for s, m, b, g in bias_results}
    rows = []
    for s, m, c, goals, median_steps in course_results:
        rows.append({"seed": s, "version": "mirrored" if m else "original", "course": c,
                     "goals": goals, "trials": TRIALS, "median_steps_to_goal": median_steps,
                     "turning_bias_100_mazes": round(bias[(s, m)][0], 3),
                     "random_maze_goal_rate": round(bias[(s, m)][1], 3)})
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in seeds:
        line = f"seed {s}:"
        for m in (False, True):
            label = "mirrored" if m else "original"
            a = next(r for r in rows if r["seed"] == s and r["version"] == label and r["course"] == "course_A")
            b = next(r for r in rows if r["seed"] == s and r["version"] == label and r["course"] == "course_B")
            line += (f"  {label}: bias {a['turning_bias_100_mazes']:+.2f}, random {a['random_maze_goal_rate']:.2f}, "
                     f"A {a['goals']}/{TRIALS}, B {b['goals']}/{TRIALS} |")
        print(line)
