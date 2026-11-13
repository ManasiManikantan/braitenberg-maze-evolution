"""Where do robots get stuck on course 2? Overlay all evaluation paths.

Usage: python plot_course2.py REPO PATHS_NPZ EPISODES_CSV OUT_PNG
"""
import csv
import sys

REPO, PATHS, EPISODES, OUT_PNG = sys.argv[1:5]
sys.path.insert(0, REPO)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from sim.maze import Maze  # noqa: E402

maze = Maze(difficulty=2)
paths = np.load(PATHS)
rows = [r for r in csv.DictReader(open(EPISODES)) if r["course"] == "2"]
fig, ax = plt.subplots(figsize=(7, 7))
for (x1, y1), (x2, y2) in maze.walls:
    ax.plot([x1, x2], [y1, y2], color="#333", lw=2.5)
for x0, y0, x1, y1 in maze.obstacles:
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#777"))
ax.add_patch(Circle(maze.goal, maze.goal_radius, color="#2a9d4a", alpha=0.5))
colors = plt.cm.tab10(np.arange(10))
ends = []
for key in paths.files:
    seed, course, _ = key.split("_")
    if course != "c2":
        continue
    p = paths[key]
    s = int(seed[4:])
    ax.plot(p[:, 0], p[:, 1], color=colors[s], lw=0.9, alpha=0.6)
    ax.plot(*p[-1], "o", color=colors[s], ms=7, mec="black")
    ends.append(p[-1])
for s in sorted({int(k.split("_")[0][4:]) for k in paths.files}):
    ax.plot([], [], color=colors[s], label=f"seed {s}")
ax.plot(0, 0, "*", color="#1f5fbf", ms=16, label="start")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(maze.x_min, maze.x_max)
ax.set_ylim(maze.y_min, maze.y_max)
ax.set_aspect("equal")
ax.set_title("Course 2: all 25 evaluation runs (5 seeds x 5 trials)\n"
             "lines = paths, circles = where each run ended")
fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
ends = np.array(ends)
print("final positions (x, y) per run:")
for e in ends:
    print("  ", np.round(e, 1))
