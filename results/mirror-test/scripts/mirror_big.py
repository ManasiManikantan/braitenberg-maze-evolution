"""Bigger mirror check: originals vs mirrors on 300 random mazes and 50 course trials.

Usage: python mirror_big.py REPO GENOMES_NPZ
"""

import sys
from multiprocessing import Pool

REPO, GENOMES = sys.argv[1:3]
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from controller.mirror import MirroredController  # noqa: E402
from evolution.experiment import course_step_limit, make_evaluation_courses, sample_random_mazes  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402

COURSES = {name: (maze, start, heading) for name, _, maze, start, heading in make_evaluation_courses()}
MAZES = sample_random_mazes(np.random.default_rng(31337), 300)


def controller_for(seed, mirrored):
    net = GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked").decode_genome(
        np.load(GENOMES)[f"seed{seed}"])
    return MirroredController(net) if mirrored else net


def episode(controller, maze, start, heading, noise_seed):
    vehicle = Vehicle(controller=controller, noise_stdev=0.05, noise_seed=noise_seed,
                      navigation_heuristics=False, start_position=start, start_heading=heading,
                      light_occlusion="blocked", **DEFAULT_ROBOT_NOISE)
    world = World(maze=maze, vehicle=vehicle)
    world.run(course_step_limit(maze, start=start))
    return int(world.reached_goal)


def maze_job(args):
    seed, mirrored, chunk = args
    controller = controller_for(seed, mirrored)
    return seed, mirrored, [episode(controller, rm.maze, rm.start, rm.start_heading, 600000 + i)
                            for i, rm in chunk]


def course_job(args):
    seed, mirrored, course = args
    controller = controller_for(seed, mirrored)
    maze, start, heading = COURSES[course]
    return seed, mirrored, course, sum(episode(controller, maze, start, heading, 510000 + t)
                                       for t in range(50))


if __name__ == "__main__":
    indexed = list(enumerate(MAZES))
    chunks = [indexed[i:i + 50] for i in range(0, 300, 50)]
    maze_jobs = [(s, m, c) for s in range(5) for m in (False, True) for c in chunks]
    course_jobs = [(s, m, c) for s in (1, 2) for m in (False, True) for c in ("course_A", "course_B")]
    with Pool(8) as pool:
        maze_results = pool.map(maze_job, maze_jobs)
        course_results = pool.map(course_job, course_jobs)
    totals = {}
    for s, m, outcomes in maze_results:
        totals.setdefault((s, m), []).extend(outcomes)
    diffs = []
    for s in range(5):
        a, b = np.array(totals[(s, False)]), np.array(totals[(s, True)])
        diffs.append(a.mean() - b.mean())
        print(f"seed {s}: 300 random mazes original {a.mean():.3f}, mirrored {b.mean():.3f}, "
              f"difference {a.mean() - b.mean():+.3f} (paired SE {np.std(a - b, ddof=1) / np.sqrt(len(a)):.3f})")
    all_a = np.concatenate([totals[(s, False)] for s in range(5)])
    all_b = np.concatenate([totals[(s, True)] for s in range(5)])
    d = all_a - all_b
    print(f"all seeds: original {all_a.mean():.3f}, mirrored {all_b.mean():.3f}, "
          f"difference {d.mean():+.3f} +/- {np.std(d, ddof=1) / np.sqrt(len(d)):.3f} (paired SE)")
    for s, m, course, goals in sorted(course_results):
        print(f"seed {s} {'mirrored' if m else 'original'} {course}: {goals}/50")
