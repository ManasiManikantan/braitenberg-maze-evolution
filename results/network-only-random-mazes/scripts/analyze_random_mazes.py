"""Tables, fixed-arc test, and plots for the random-maze run.

Usage: python analyze_random_mazes.py REPO OUT_DIR TP_GENOMES PLOT_SEED_A PLOT_SEED_B TP_SEED
"""

import csv
import itertools
import math
import sys

REPO, OUT, TP_GENOMES = sys.argv[1:4]
SEED_A, SEED_B, TP_SEED = sys.argv[4:7]
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from evolution.experiment import course_step_limit  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.maze_generator import generate_perfect_maze  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, DEFAULT_SIDE_SENSOR_ANGLE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402

rows = list(csv.DictReader(open(f"{OUT}/random_mazes_test_episodes.csv")))
for r in rows:
    for k in ("goal_success", "path_progress", "collisions", "steps", "longest_pinned",
              "pct_moving_with_wall_beside", "wall_side_left_share"):
        r[k] = float(r[k])


def summarize(selected):
    n = len(selected)
    shares = [r["wall_side_left_share"] for r in selected if not math.isnan(r["wall_side_left_share"])]
    return {
        "runs": n,
        "goal_success": round(sum(r["goal_success"] for r in selected) / n, 3),
        "mean_path_progress": round(np.mean([r["path_progress"] for r in selected]), 3),
        "pct_steps_in_collision": round(100 * sum(r["collisions"] for r in selected)
                                        / sum(r["steps"] for r in selected), 1),
        "longest_pinned_mean": round(np.mean([r["longest_pinned"] for r in selected]), 1),
        "pct_moving_with_wall_beside": round(np.mean([r["pct_moving_with_wall_beside"] for r in selected]), 1),
        "wall_on_left_share": round(float(np.mean(shares)), 2) if shares else "",
    }


table = []
for controller in ("time_pressure", "random_mazes"):
    for group, course in [("random_test", None), ("courses", "1"), ("courses", "2"),
                          ("courses", "3"), ("courses", "4"), ("courses", None)]:
        sel = [r for r in rows if r["controller"] == controller and r["set"] == group
               and (course is None or r["course"] == course)]
        label = "random test mazes (50)" if group == "random_test" else (
            f"course {course}" if course else "courses 1-4")
        table.append({"controller": controller, "test_set": label, **summarize(sel)})
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)
for t in table:
    print(t)

print("\nper seed (random test mazes goal success | courses goal success per course):")
SEEDS = {c: sorted({int(r["seed"]) for r in rows if r["controller"] == c})
         for c in ("time_pressure", "random_mazes")}
for controller in ("time_pressure", "random_mazes"):
    for s in SEEDS[controller]:
        rt = [r for r in rows if r["controller"] == controller and r["seed"] == str(s) and r["set"] == "random_test"]
        cs = [r for r in rows if r["controller"] == controller and r["seed"] == str(s) and r["set"] == "courses"]
        per = " ".join(f"c{c}:{int(sum(r['goal_success'] for r in cs if r['course'] == c))}/5" for c in "1234")
        shares = [r["wall_side_left_share"] for r in rt if not math.isnan(r["wall_side_left_share"])]
        print(f"  {controller:13} seed{s}: random {sum(r['goal_success'] for r in rt):.0f}/50 "
              f"prog {np.mean([r['path_progress'] for r in rt]):+.2f} | {per} | "
              f"wall beside {np.mean([r['pct_moving_with_wall_beside'] for r in rt]):.0f}% "
              f"left-share {np.mean(shares):.2f}")

# ---- fixed-arc test: spread of start-frame trajectories across mazes ----
early = np.load(f"{OUT}/early_trajectories.npz")
STEP = 150


def spread(keys):
    points = [early[k][STEP] for k in keys if len(early[k]) > STEP]
    if len(points) < 2:
        return float("nan"), len(points)
    return float(np.mean([math.dist(a, b) for a, b in itertools.combinations(points, 2)])), len(points)


ga = GeneticAlgorithm(population_size=4, hidden_size=4, seed=0)
genomes = {"random_mazes": np.load(f"{OUT}/random_mazes_genomes.npz"), "time_pressure": np.load(TP_GENOMES)}
side = {"random_mazes": DEFAULT_SIDE_SENSOR_ANGLE, "time_pressure": 0.0}
test_mazes = [generate_perfect_maze(rng) for rng in [np.random.default_rng(2026)] for _ in range(50)]


def run(controller, maze, start, heading, noise_seed, side_angle, steps=None):
    vehicle = Vehicle(controller=controller, noise_stdev=0.05, noise_seed=noise_seed,
                      navigation_heuristics=False, start_position=start, start_heading=heading,
                      side_sensor_angle=side_angle, **DEFAULT_ROBOT_NOISE)
    world = World(maze=maze, vehicle=vehicle)
    limit = steps or min(3000, course_step_limit(maze, start=start))
    collided = []
    while world.steps < limit and not world.reached_goal:
        before = world.collisions
        world.step()
        if world.collisions > before:
            collided.append(world.position_history[-1])
    return world, np.array(world.position_history), collided


def steering_stats(net, side_angle, mazes):
    """Turn command (left - right motor) on free-driving steps across mazes."""
    turns = []
    for idx, rm in enumerate(mazes):
        vehicle = Vehicle(controller=net, noise_stdev=0.05, noise_seed=4000 + idx,
                          navigation_heuristics=False, start_position=rm.start,
                          start_heading=rm.start_heading, side_sensor_angle=side_angle,
                          **DEFAULT_ROBOT_NOISE)
        world = World(maze=rm.maze, vehicle=vehicle)
        limit = min(3000, course_step_limit(rm.maze, start=rm.start))
        while world.steps < limit and not world.reached_goal:
            before = world.collisions
            world.step()
            if world.collisions == before:
                turns.append(vehicle.left_motor - vehicle.right_motor)
    turns = np.array(turns)
    dominant = np.sign(np.median(turns)) or 1.0
    return float(np.std(turns)), float(np.mean(np.sign(turns) == -dominant)), float(np.median(turns))


print("\nsteering test (free-driving steps on 20 random test mazes): a fixed arc has a nearly")
print("constant turn command; a maze-dependent controller varies it and turns both ways")
for controller in ("time_pressure", "random_mazes"):
    for s in SEEDS[controller]:
        net = ga.decode_genome(genomes[controller][f"seed{s}"])
        std, against, median = steering_stats(net, side[controller], test_mazes[:20])
        print(f"  {controller:13} seed{s}: median turn {median:+.2f}  std {std:.2f}  "
              f"steps turning against usual direction {100 * against:.0f}%")

print(f"\nfixed-arc test: mean pairwise distance between start-frame positions at step {STEP}")
print("  (across the 50 random test mazes vs. same maze/start with 20 noise seeds = noise floor)")
for controller in ("time_pressure", "random_mazes"):
    for s in SEEDS[controller]:
        keys = [k for k in early.files if k.startswith(f"{controller}_seed{s}_random_test")]
        across, n = spread(keys)
        net = ga.decode_genome(genomes[controller][f"seed{s}"])
        rm = test_mazes[0]
        # noise floor: same maze and nominal start, only the noise seed changes
        floor_pts = []
        for ns in range(20):
            _, path, _ = run(net, rm.maze, rm.start, rm.start_heading, 9000 + ns, side[controller], STEP + 1)
            if len(path) > STEP:
                rel = path[STEP] - path[0]
                c, s_ = math.cos(-rm.start_heading), math.sin(-rm.start_heading)
                floor_pts.append((rel[0] * c - rel[1] * s_, rel[0] * s_ + rel[1] * c))
        floor = float(np.mean([math.dist(a, b) for a, b in itertools.combinations(floor_pts, 2)])) \
            if len(floor_pts) > 1 else float("nan")
        print(f"  {controller:13} seed{s}: across mazes {across:5.2f} (n={n})   noise floor {floor:5.2f}")


# ---- plots ----
def draw(ax, maze, path, collided, title):
    for (x1, y1), (x2, y2) in maze.walls:
        ax.plot([x1, x2], [y1, y2], color="#333", lw=2)
    for x0, y0, x1, y1 in maze.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#777"))
    ax.add_patch(Circle(maze.goal, maze.goal_radius, color="#2a9d4a", alpha=0.5))
    sc = ax.scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), cmap="viridis", s=2)
    ax.plot(*path[0], "o", color="#1f5fbf", ms=8)
    if collided:
        c = np.array(collided)
        ax.plot(c[:, 0], c[:, 1], "x", color="#d62728", ms=5)
    ax.set_xlim(maze.x_min, maze.x_max)
    ax.set_ylim(maze.y_min, maze.y_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return sc


def maze_title(name, world, collided):
    pct = 100 * world.collisions / max(world.steps, 1)
    return f"{name}: {'GOAL' if world.reached_goal else 'no goal'} in {world.steps} steps\n{pct:.0f}% steps in collision"


def pick_two_mazes(seed):
    sel = [r for r in rows if r["controller"] == "random_mazes" and r["seed"] == seed and r["set"] == "random_test"]
    goals = [int(r["course"][1:]) for r in sel if r["goal_success"]]
    fails = [int(r["course"][1:]) for r in sel if not r["goal_success"]]
    return (goals[:2] if len(goals) >= 2 else goals + fails[:2 - len(goals)])


for seed in (SEED_A, SEED_B):
    net = ga.decode_genome(genomes["random_mazes"][f"seed{seed}"])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    panels = []
    for idx in pick_two_mazes(seed):
        rm = test_mazes[idx]
        world, path, collided = run(net, rm.maze, rm.start, rm.start_heading,
                                    int(seed) * 100000 + idx + 700000, DEFAULT_SIDE_SENSOR_ANGLE)
        panels.append((rm.maze, path, collided, maze_title(f"random test maze {idx}", world, collided)))
    for d in (1, 2):
        maze = Maze(difficulty=d)
        world, path, collided = run(net, maze, (0.0, 0.0), 0.0, int(seed) * 1000 + d * 10 + 500000,
                                    DEFAULT_SIDE_SENSOR_ANGLE)
        panels.append((maze, path, collided, maze_title(f"course {d} (held out)", world, collided)))
    for ax, panel in zip(axes, panels):
        sc = draw(ax, *panel)
    fig.colorbar(sc, ax=axes, fraction=0.012, label="step")
    fig.suptitle(f"Random-maze-trained controller (seed {seed}), 60-degree side sensors; "
                 "blue = start, green = goal, red x = collision")
    fig.savefig(f"{OUT}/trajectories_random_mazes_seed{seed}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for ax, (controller, seed, title) in zip(axes, [
        ("time_pressure", TP_SEED, f"time-pressure controller (seed {TP_SEED}), trained on courses 1-4"),
        ("random_mazes", SEED_A, f"random-maze controller (seed {SEED_A})")]):
    keys = [k for k in early.files if k.startswith(f"{controller}_seed{seed}_random_test")]
    for k in keys:
        p = early[k]
        ax.plot(p[:, 0], p[:, 1], lw=0.8, alpha=0.6)
    ax.plot(0, 0, "o", color="#1f5fbf", ms=8)
    ax.arrow(0, 0, 1.5, 0, width=0.12, color="k")
    ax.set_aspect("equal")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.grid(alpha=0.3)
    ax.set_title(f"{title}\nfirst {early[keys[0]].shape[0] - 1} steps on 50 random test mazes, "
                 "in the robot's start frame", fontsize=9)
fig.suptitle("Fixed-arc test: overlapping paths = same route regardless of maze")
fig.savefig(f"{OUT}/start_frame_overlay.png", dpi=110, bbox_inches="tight")
print("plots written")
