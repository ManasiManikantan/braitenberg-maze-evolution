"""Contact entries (new collisions) per run on 50 calibration mazes (seed 31), no health limit.
Usage: python calibrate_health.py REPO GA_GENOMES_NPZ"""
import sys
REPO, GENOMES = sys.argv[1:3]
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
        r = record_run(c, m, s, h, noise_seed=n)
        out.append((r.world.contact_entries, r.world.reached_goal, r.steps))
    return name, out

if __name__ == "__main__":
    g = np.load(GENOMES)
    specs = [("wall left", "left"), ("wall right", "right")] + [(f"seed{s}", g[f"seed{s}"]) for s in (3, 19, 2, 13)]
    with Pool(3) as p:
        res = p.map(job, specs)
    for name, out in res:
        e = np.array([o[0] for o in out])
        print(f"{name:10} entries median {np.median(e):.0f} p75 {np.percentile(e,75):.0f} p90 {np.percentile(e,90):.0f} max {e.max()} | "
              + " ".join(f"H={h}:{np.mean(e < h):.2f}" for h in (5, 10, 15, 20, 25, 30, 40)))
