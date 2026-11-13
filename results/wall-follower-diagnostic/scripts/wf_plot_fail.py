import sys, csv
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import wall_follower_diag as d
from wall_follower import WallFollower
from sim.robot import Vehicle, DEFAULT_ROBOT_NOISE
from sim.world import World
OUT = sys.argv[3]
rows = list(csv.DictReader(open(f"{OUT}/wall_follower_runs.csv")))
sets = d.test_sets()
picks = []
for set_name, kinds in (("random_20x20", ("stuck",)), ("random_20x20", ("loop",)), ("course_2", ("stuck",)),
                        ("course_1", ("stuck",)), ("course_3", ("stuck",)), ("course_4", ("stuck",))):
    for r in rows:
        if r["set"] == set_name and r["failure_no_limit"].split(" at")[0] in kinds:
            picks.append(r); break
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, r in zip(axes.ravel(), picks):
    entries = sets[r["set"]]
    idx = [e[0] for e in entries].index(r["maze"])
    label, maze, start, heading, _ = entries[idx]
    side = r["controller"].split("_")[0]
    noise_seed = 31000 + idx
    v = Vehicle(controller=WallFollower(side), noise_stdev=0.05, noise_seed=noise_seed, navigation_heuristics=False,
                start_position=start, start_heading=heading, **DEFAULT_ROBOT_NOISE)
    w = World(maze=maze, vehicle=v); w.run(d.CAP)
    p = np.array(w.position_history)
    for (a, b), (c, e) in maze.walls: ax.plot([a, c], [b, e], color="#333", lw=2)
    for x0, y0, x1, y1 in maze.obstacles: ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#777"))
    ax.add_patch(Circle(maze.goal, maze.goal_radius, color="#2a9d4a", alpha=.5))
    ax.scatter(p[:, 0], p[:, 1], c=np.arange(len(p)), s=1, cmap="viridis")
    ax.plot(*p[0], "o", color="#1f5fbf", ms=8); ax.plot(*p[-1], "s", color="#d62728", ms=8)
    ax.set_aspect("equal"); ax.set_xlim(maze.x_min, maze.x_max); ax.set_ylim(maze.y_min, maze.y_max)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{r['set']} {r['maze']}, {r['controller']}: {r['failure_no_limit']}\n"
                 f"{w.steps} steps, {100 * w.collisions / w.steps:.0f}% in collision; red square = end", fontsize=9)
    print(r["set"], r["maze"], r["controller"], r["failure_no_limit"], "end", r["end_x"], r["end_y"], "width", r["end_passage_width"])
fig.suptitle("Wall-follower failures with no time limit (blue = start, green = goal)")
fig.savefig(f"{OUT}/wall_follower_failures.png", dpi=90, bbox_inches="tight")
