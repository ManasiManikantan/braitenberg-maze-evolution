"""Battery calibration (energy used per unit shortest-route length on successful runs) and
distance-sensor distributions, 30 calibration mazes (seed 31). Usage: calibrate_zoo.py REPO GA_NPZ"""
import sys
REPO, GENOMES = sys.argv[1:3]
sys.path.insert(0, REPO)
from multiprocessing import Pool
import numpy as np
from controller.wall_follower import WallFollower
from evolution.genetic_algorithm import GeneticAlgorithm
from evolution.metrics import random_maze_episodes, record_run
from evolution.fitness import path_distances

EPISODES = random_maze_episodes(30, seed=31, noise_seed_base=310000)

def job(spec):
    name, payload = spec
    c = WallFollower(payload) if name.startswith("wall") else \
        GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked").decode_genome(payload)
    ratios, sensors = [], []
    for m, s, h, n in EPISODES:
        r = record_run(c, m, s, h, noise_seed=n, on_step=lambda v: sensors.append(v.controller_inputs()[:3]))
        if r.reached_goal:
            ratios.append(r.world.energy_used / float(path_distances(m, [s], r.radius)[0]))
    return name, np.array(ratios), np.array(sensors)

if __name__ == "__main__":
    g = np.load(GENOMES)
    specs = [("wall left", "left"), ("wall right", "right"), ("seed3", g["seed3"]), ("seed2", g["seed2"])]
    with Pool(2) as p:
        res = p.map(job, specs)
    for name, r, sens in res:
        print(f"{name:10} success {len(r)}/30 energy per route unit: median {np.median(r):.1f} p75 {np.percentile(r,75):.1f} "
              f"p90 {np.percentile(r,90):.1f} | budget fit c=20:{np.mean(r<20):.2f} c=30:{np.mean(r<30):.2f} c=40:{np.mean(r<40):.2f} c=60:{np.mean(r<60):.2f}")
        print(f"           sensors L/F/R percentiles 50/90/99: " + " ".join(
            f"{np.percentile(sens[:, i], 50):.2f}/{np.percentile(sens[:, i], 90):.2f}/{np.percentile(sens[:, i], 99):.2f}" for i in range(3)))
