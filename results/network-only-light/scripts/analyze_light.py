"""Tables, training curves, light-wiring analysis, and plots for the light runs.

Usage: python analyze_light.py REPO OUT_DIR PREVIOUS_RANDOM_MAZES_EPISODES_CSV
"""

import csv
import math
import sys

REPO, OUT, PREV = sys.argv[1:4]
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from evolution.experiment import course_step_limit, curriculum_maze_options  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.maze import Maze  # noqa: E402
from sim.maze_generator import generate_perfect_maze  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402

MODES = ("blocked", "through")
NUMERIC = ("goal_success", "path_progress", "collisions", "steps", "longest_pinned",
           "pct_moving_with_wall_beside")


def load(path, rename=None):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in NUMERIC:
            r[k] = float(r[k])
        if rename and r["controller"] in rename:
            r["controller"] = rename[r["controller"]]
    return rows


rows = []
for mode in MODES:
    rows += load(f"{OUT}/light_{mode}_test_episodes.csv")
rows += [r for r in load(PREV, {"random_mazes": "previous_no_light"})
         if r["controller"] == "previous_no_light"]


def summarize(sel):
    n = len(sel)
    return {
        "runs": n,
        "goal_success": round(sum(r["goal_success"] for r in sel) / n, 3),
        "mean_path_progress": round(float(np.mean([r["path_progress"] for r in sel])), 3),
        "pct_steps_in_collision": round(100 * sum(r["collisions"] for r in sel) / sum(r["steps"] for r in sel), 1),
        "longest_pinned_mean": round(float(np.mean([r["longest_pinned"] for r in sel])), 1),
    }


CONTROLLERS = ("light_blocked", "untrained_blocked", "light_through", "untrained_through",
               "previous_no_light")
table = []
for controller in CONTROLLERS:
    for group, course in [("random_test", None), ("courses", "1"), ("courses", "2"),
                          ("courses", "3"), ("courses", "4"), ("courses", None)]:
        sel = [r for r in rows if r["controller"] == controller and r["set"] == group
               and (course is None or r["course"] == course)]
        if not sel:
            continue
        label = "random test mazes (50)" if group == "random_test" else (
            f"course {course}" if course else "courses 1-4")
        table.append({"controller": controller, "test_set": label, **summarize(sel)})
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)
print("SUMMARY")
for t in table:
    print("  " + "  ".join(f"{k}={v}" for k, v in t.items()))

print("\nPER SEED (random test goals / 50 | course goals / 5)")
for controller in CONTROLLERS:
    for s in range(5):
        rt = [r for r in rows if r["controller"] == controller and r["seed"] == str(s) and r["set"] == "random_test"]
        cs = [r for r in rows if r["controller"] == controller and r["seed"] == str(s) and r["set"] == "courses"]
        if not rt:
            continue
        per = " ".join(f"c{c}:{int(sum(r['goal_success'] for r in cs if r['course'] == c))}" for c in "1234")
        print(f"  {controller:18} seed{s}: random {int(sum(r['goal_success'] for r in rt)):2d}/50 "
              f"prog {np.mean([r['path_progress'] for r in rt]):+.2f} coll "
              f"{100 * sum(r['collisions'] for r in rt) / sum(r['steps'] for r in rt):.0f}% | {per}")

# ---- training curves ----
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
print("\nTRAINING CURVES (median over seeds; population goal rate on its training mazes)")
for col, mode in enumerate(MODES):
    hist = list(csv.DictReader(open(f"{OUT}/light_{mode}_history.csv")))
    gens = sorted({int(h["generation"]) for h in hist})
    for row_index, key, label in ((0, "mean_goal_rate", "population goal rate"),
                                  (1, "mean_fitness", "population mean fitness")):
        ax = axes[row_index, col]
        series = {}
        for s in sorted({int(h["seed"]) for h in hist}):
            ys = [float(h[key]) for h in hist if h["seed"] == str(s)]
            series[s] = ys
            ax.plot(gens, ys, lw=0.8, alpha=0.5)
        median = np.median(np.array(list(series.values())), axis=0)
        ax.plot(gens, median, color="black", lw=2, label="median of 5 seeds")
        for boundary in (20, 40):
            ax.axvline(boundary - 0.5, color="#888", ls="--", lw=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if row_index == 0:
            ax.set_title(f"light occlusion: {mode}\n(12x12 | 16x16 | 20x20 mazes)", fontsize=10)
        if key == "mean_goal_rate" and len(gens) >= 60:
            stages = [(0, 20), (20, 40), (40, 60)]
            text = "  ".join(
                f"gens {a}-{b - 1}: {np.mean(median[a:a + 5]):.2f} -> {np.mean(median[b - 5:b]):.2f}"
                for a, b in stages)
            print(f"  {mode:8} goal rate (first 5 -> last 5 gens of each stage): {text}")
    axes[1, col].set_xlabel("generation")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Training curves (fitness is not comparable across curriculum stages: mazes get larger)")
fig.savefig(f"{OUT}/training_curves.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ---- light wiring ----
print("\nLIGHT WIRING: mean partial derivatives of motor outputs w.r.t. light inputs over")
print("states visited on 20 test mazes. Simulator convention: heading += k (L - R), so a")
print("larger LEFT motor turns the robot LEFT (opposite to a physical differential drive).")
wiring_rows = []
for mode in MODES:
    genomes = np.load(f"{OUT}/light_{mode}_genomes.npz")
    visited = np.load(f"{OUT}/light_{mode}_visited_inputs.npz")
    ga = GeneticAlgorithm(population_size=4, hidden_size=4, seed=0, light_occlusion=mode)
    for key in sorted(genomes.files):
        net = ga.decode_genome(genomes[key])
        states = visited[key]
        eps = 1e-4
        jac = np.zeros((2, 2))
        for state in states:
            base = net.forward(state)
            for j, idx in enumerate((3, 4)):
                bumped = np.array(state, dtype=float)
                bumped[idx] += eps
                jac[:, j] += (net.forward(bumped) - base) / eps
        jac /= max(len(states), 1)
        dl_ll, dl_lr = jac[0]
        dr_ll, dr_lr = jac[1]
        same = abs(dl_ll) + abs(dr_lr)
        crossed = abs(dl_lr) + abs(dr_ll)
        if same >= crossed:
            topology = "same-side"
            sign = "excitatory" if dl_ll + dr_lr > 0 else "inhibitory"
        else:
            topology = "crossed"
            sign = "excitatory" if dl_lr + dr_ll > 0 else "inhibitory"
        # turn response to light on the LEFT vs RIGHT: + means L-R grows with left light => turns left
        turn = (dl_ll - dr_ll) - (dl_lr - dr_lr)
        speed = (dl_ll + dr_ll) + (dl_lr + dr_lr)
        toward = turn > 0
        behaviour = {(True, True): "aggression (2b-like)", (True, False): "love (3a-like)",
                     (False, True): "fear (2a-like)", (False, False): "explorer (3b-like)"}[(toward, speed > 0)]
        mean_light = float(np.mean(states[:, 3:5])) if len(states) else float("nan")
        lit_share = float(np.mean(states[:, 3:5].max(axis=1) > 0.02)) if len(states) else float("nan")
        wiring_rows.append({
            "mode": mode, "seed": key, "dL/dlightL": round(dl_ll, 3), "dL/dlightR": round(dl_lr, 3),
            "dR/dlightL": round(dr_ll, 3), "dR/dlightR": round(dr_lr, 3),
            "code_wiring": f"{topology} {sign}", "turns": "toward light" if toward else "away from light",
            "speed_with_light": "faster" if speed > 0 else "slower", "behaves_like": behaviour,
            "turn_strength": round(turn, 3), "share_of_states_with_light": round(lit_share, 2),
            "mean_light_reading": round(mean_light, 3),
        })
        print(f"  {mode:8} {key}: [dL/dlL {dl_ll:+.2f}, dL/dlR {dl_lr:+.2f}; dR/dlL {dr_ll:+.2f}, "
              f"dR/dlR {dr_lr:+.2f}]  code: {topology} {sign}; turns "
              f"{'TOWARD' if toward else 'AWAY from'} light (strength {turn:+.2f}), "
              f"{'faster' if speed > 0 else 'slower'} in light -> {behaviour}; light seen in "
              f"{100 * lit_share:.0f}% of states")
with open(f"{OUT}/light_wiring.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(wiring_rows[0]))
    w.writeheader()
    w.writerows(wiring_rows)

# ---- trajectory plots: one robot per mode on two random test mazes + courses 1, 2 ----
test_mazes = [generate_perfect_maze(rng) for rng in [np.random.default_rng(2026)] for _ in range(50)]


def run(net, mode, maze, start, heading, noise_seed):
    vehicle = Vehicle(controller=net, noise_stdev=0.05, noise_seed=noise_seed,
                      navigation_heuristics=False, start_position=start, start_heading=heading,
                      light_occlusion=mode, **DEFAULT_ROBOT_NOISE)
    world = World(maze=maze, vehicle=vehicle)
    limit = min(3000, course_step_limit(maze, start=start))
    collided = []
    while world.steps < limit and not world.reached_goal:
        before = world.collisions
        world.step()
        if world.collisions > before:
            collided.append(world.position_history[-1])
    return world, np.array(world.position_history), collided


def draw(ax, maze, path, collided, title):
    for (x1, y1), (x2, y2) in maze.walls:
        ax.plot([x1, x2], [y1, y2], color="#333", lw=2)
    for x0, y0, x1, y1 in maze.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#777"))
    ax.add_patch(Circle(maze.goal, maze.goal_radius, color="#e6b800", alpha=0.6))
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


for mode in MODES:
    sel = [r for r in rows if r["controller"] == f"light_{mode}" and r["set"] == "random_test"]
    by_seed = {s: sum(r["goal_success"] for r in sel if r["seed"] == str(s))
               for s in sorted({int(r["seed"]) for r in sel})}
    best_seed = max(by_seed, key=by_seed.get)
    mine = [r for r in sel if r["seed"] == str(best_seed)]
    goals = [int(r["course"][1:]) for r in mine if r["goal_success"]]
    fails = [int(r["course"][1:]) for r in mine if not r["goal_success"]]
    # two different mazes: prefer two successes with different route lengths
    picks = (goals[:1] + [g for g in goals[1:] if test_mazes[g].route_cells != test_mazes[goals[0]].route_cells][:1]) if goals else []
    picks = (picks + goals + fails)[:2] if len(picks) < 2 else picks
    net = GeneticAlgorithm(population_size=4, hidden_size=4, seed=0, light_occlusion=mode).decode_genome(
        np.load(f"{OUT}/light_{mode}_genomes.npz")[f"seed{best_seed}"])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    panels = []
    for idx in picks:
        rm = test_mazes[idx]
        world, path, collided = run(net, mode, rm.maze, rm.start, rm.start_heading,
                                    best_seed * 100000 + idx + 700000)
        panels.append((rm.maze, path, collided,
                       f"random test maze {idx} ({rm.route_cells}-cell route): "
                       f"{'GOAL' if world.reached_goal else 'no goal'} in {world.steps}\n"
                       f"{100 * world.collisions / max(world.steps, 1):.0f}% steps in collision"))
    for d in (1, 2):
        maze = Maze(difficulty=d)
        world, path, collided = run(net, mode, maze, (0.0, 0.0), 0.0, best_seed * 1000 + d * 10 + 500000)
        panels.append((maze, path, collided,
                       f"course {d} (held out): {'GOAL' if world.reached_goal else 'no goal'} in "
                       f"{world.steps}\n{100 * world.collisions / max(world.steps, 1):.0f}% steps in collision"))
    for ax, panel in zip(axes, panels):
        sc = draw(ax, *panel)
    fig.colorbar(sc, ax=axes, fraction=0.012, label="step")
    fig.suptitle(f"Light-sensing controller, occlusion '{mode}' (seed {best_seed}); blue = start, "
                 "yellow = goal light, red x = collision")
    fig.savefig(f"{OUT}/trajectories_light_{mode}_seed{best_seed}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"plot: trajectories_light_{mode}_seed{best_seed}.png (mazes {picks})")
