import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from evolution.genetic_algorithm import GeneticAlgorithm
from evolution.experiment import make_evaluation_courses, course_step_limit
from sim.robot import Vehicle, DEFAULT_ROBOT_NOISE
from sim.world import World
genomes = np.load(sys.argv[2]); out = sys.argv[3]
ga = GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked")
courses = {n: (m, s, h) for n, _, m, s, h in make_evaluation_courses()}
seeds = [int(x) for x in sys.argv[4:]]
fig, axes = plt.subplots(len(seeds), 2, figsize=(8, 4 * len(seeds)))
for row, seed in zip(np.atleast_2d(axes), seeds):
    net = ga.decode_genome(genomes[f"seed{seed}"])
    for ax, name in zip(row, ("course_A", "course_B")):
        m, s, h = courses[name]
        for t in range(3):
            v = Vehicle(controller=net, noise_stdev=0.05, noise_seed=500000 + t, navigation_heuristics=False,
                        start_position=s, start_heading=h, light_occlusion="blocked", **DEFAULT_ROBOT_NOISE)
            w = World(maze=m, vehicle=v); w.run(course_step_limit(m, start=s))
            p = np.array(w.position_history)
            ax.plot(p[:, 0], p[:, 1], lw=1, label=f"trial {t}: {'goal' if w.reached_goal else 'no'} {w.steps}")
        for (a, b), (c, d) in m.walls: ax.plot([a, c], [b, d], color="#333", lw=2)
        ax.add_patch(Circle(m.goal, 1, color="#e6b800", alpha=.6)); ax.plot(*s, "o", color="#1f5fbf", ms=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7, loc="center")
        ax.set_title(f"seed {seed} on {name}", fontsize=9)
fig.savefig(out, dpi=90, bbox_inches="tight")
