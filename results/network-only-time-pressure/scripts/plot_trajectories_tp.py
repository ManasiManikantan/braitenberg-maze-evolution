"""Plot trajectories of evolved controllers on all four courses + cheat diagnostics.

Usage: python plot_trajectories.py REPO GENOMES_NPZ OUT_DIR KEY [KEY ...]
"""

import math
import sys

REPO, GENOMES, OUT, KEYS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from evolution.experiment import course_step_limit  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402

STEPS = 3000
genomes = np.load(GENOMES)
ga = GeneticAlgorithm(population_size=4, hidden_size=4, seed=0)


def run(controller, difficulty, noise_seed):
    maze = Maze(difficulty=difficulty)
    world = World(maze=maze, vehicle=Vehicle(
        controller=controller, noise_stdev=0.05, noise_seed=noise_seed,
        navigation_heuristics=False, **DEFAULT_ROBOT_NOISE))
    headings, collided = [world.vehicle.orientation], []
    limit = min(STEPS, course_step_limit(maze) or STEPS)
    flags = []
    while world.steps < limit and not world.reached_goal:
        before = world.collisions
        world.step()
        headings.append(world.vehicle.orientation)
        flags.append(world.collisions > before)
        if world.collisions > before:
            collided.append(world.position_history[-1])
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    pos = np.array(world.position_history)
    path_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    turning = float(np.sum(np.abs(np.diff(headings))))
    contact = np.mean([maze.clearance(p, world.vehicle.radius) < 0.05 for p in pos[::3]])
    return world, pos, collided, {
        "goal": world.reached_goal, "steps": world.steps, "collisions": world.collisions,
        "path_len": path_len, "net_disp": math.dist(pos[0], pos[-1]),
        "turn_per_unit": turning / max(path_len, 1e-9),
        "wall_contact_frac": float(contact), "limit": limit,
        "pct_collision": 100.0 * world.collisions / max(world.steps, 1), "longest_pinned": longest,
    }


for key in KEYS:
    controller = ga.decode_genome(genomes[key])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    print(f"== {key}  bias_output={np.round(controller.bias_output, 2)}")
    for ax, d in zip(axes, (1, 2, 3, 4)):
        world, pos, collided, stats = run(controller, d, noise_seed=777 + d)
        maze = world.maze
        for (x1, y1), (x2, y2) in maze.walls:
            ax.plot([x1, x2], [y1, y2], color="#333", lw=2)
        for x0, y0, x1, y1 in maze.obstacles:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#777"))
        ax.add_patch(Circle(maze.goal, maze.goal_radius, color="#2a9d4a", alpha=0.5))
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=np.arange(len(pos)), cmap="viridis", s=2)
        ax.plot(*pos[0], "o", color="#1f5fbf", ms=8)
        if collided:
            c = np.array(collided)
            ax.plot(c[:, 0], c[:, 1], "x", color="#d62728", ms=5)
        ax.set_xlim(maze.x_min, maze.x_max)
        ax.set_ylim(maze.y_min, maze.y_max)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"course {d}: {'GOAL' if stats['goal'] else 'no goal'} in {stats['steps']}/{stats['limit']} steps\n"
                     f"{stats['pct_collision']:.0f}% steps in collision, longest pin {stats['longest_pinned']}", fontsize=9)
        print(f"  course {d}: " + ", ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
    fig.colorbar(sc, ax=axes, fraction=0.012, label="step")
    fig.suptitle(f"Evolved network-only controller, per-course time limits ({key}); blue = start, green = goal, red x = collision")
    fig.savefig(f"{OUT}/trajectories_{key}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
