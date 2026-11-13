"""Survival under health H with 1 per new hit + c per contact step, from per-run counts
(50 calibration mazes, seed 31, no health). Usage: calibrate_contact.py REPO GA_NPZ FEAR_NPZ"""
import sys
REPO, GENOMES, FEAR = sys.argv[1:4]
sys.path.insert(0, REPO)
from multiprocessing import Pool
import numpy as np
from controller.wall_follower import WallFollower
from evolution.genetic_algorithm import GeneticAlgorithm
from evolution.metrics import random_maze_episodes, record_run

EPISODES = random_maze_episodes(50, seed=31, noise_seed_base=310000)

def job(spec):
    name, payload = spec
    c = WallFollower(payload) if name.startswith("wall") else \
        GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked").decode_genome(payload)
    out = []
    for m, s, h, n in EPISODES:
        w = record_run(c, m, s, h, noise_seed=n).world
        out.append((w.contact_entries, w.collisions, w.reached_goal))
    return name, np.array(out, dtype=float)

if __name__ == "__main__":
    g = np.load(GENOMES)
    specs = [("wall left", "left"), ("wall right", "right"), ("seed3", g["seed3"]), ("seed2", g["seed2"]),
             ("fear", np.load(FEAR)["fear"])]
    with Pool(3) as p:
        res = p.map(job, specs)
    for name, a in res:
        hits, steps, goal = a.T
        line = f"{name:10} hits med {np.median(hits):.0f} contact steps med {np.median(steps):.0f} p75 {np.percentile(steps,75):.0f} goal {goal.mean():.2f} |"
        for H in (15, 20):
            for c in (0.0, 0.02, 0.05, 0.1):
                line += f" H{H} c{c}:{np.mean(hits + c * steps < H):.2f}"
        print(line)
