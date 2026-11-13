"""Tables, training curves, turning bias, light wiring, and plots for the simple setup.

Physical motor convention: heading += k (R - L), so R > L turns LEFT.
Usage: python analyze_simple.py REPO OUT_DIR WALL_FOLLOWER_CSV
"""

import csv
import sys

REPO, OUT, WF = sys.argv[1:4]
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from evolution.experiment import course_step_limit, make_evaluation_courses, sample_random_mazes  # noqa: E402
from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle  # noqa: E402
from sim.world import World  # noqa: E402

SETS = ("random_12x12", "course_A", "course_B", "course_C")
rows = list(csv.DictReader(open(f"{OUT}/test_episodes.csv")))
wf = list(csv.DictReader(open(WF)))


def pct(values):
    return round(100 * float(np.mean(values)), 1) if values else ""


table = []
for controller in ("evolved", "untrained", "left_wall", "right_wall"):
    for set_name in SETS:
        if controller in ("evolved", "untrained"):
            sel = [r for r in rows if r["controller"] == controller and r["set"] == set_name]
            entry = {
                "controller": controller, "test_set": set_name, "runs": len(sel),
                "goal_success_pct": pct([float(r["goal_success"]) for r in sel]),
                "mean_path_progress": round(float(np.mean([float(r["path_progress"]) for r in sel])), 3),
                "pct_steps_in_collision": round(100 * sum(float(r["collisions"]) for r in sel)
                                                / sum(float(r["steps"]) for r in sel), 1),
            }
        else:
            sel = [r for r in wf if r["controller"] == controller and r["set"] == set_name]
            entry = {
                "controller": controller, "test_set": set_name, "runs": len(sel),
                "goal_success_pct": pct([float(r["goal_within_limit"]) for r in sel]),
                "mean_path_progress": "",
                "pct_steps_in_collision": round(float(np.mean([float(r["pct_steps_in_collision"]) for r in sel])), 1),
            }
        table.append(entry)
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)
print("SUMMARY (goal success within the 8x limit)")
for t in table:
    print("  " + "  ".join(f"{k}={v}" for k, v in t.items()))

seeds = sorted({int(r["seed"]) for r in rows if r["controller"] == "evolved"})
print("\nPER SEED (evolved): random 12x12 goals/50 | A, B, C goals/10")
per_seed = {}
for s in seeds:
    sel = [r for r in rows if r["controller"] == "evolved" and r["seed"] == str(s)]
    counts = {n: int(sum(float(r["goal_success"]) for r in sel if r["set"] == n)) for n in SETS}
    per_seed[s] = counts
    print(f"  seed{s}: random {counts['random_12x12']}/50 | A {counts['course_A']}/10 "
          f"B {counts['course_B']}/10 C {counts['course_C']}/10")

# ---- training curves ----
hist = list(csv.DictReader(open(f"{OUT}/history.csv")))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
print("\nTRAINING CURVES (median over seeds, mean of gens)")
for ax, key, label in ((axes[0], "mean_goal_rate", "population goal rate (shared 8 mazes)"),
                       (axes[1], "mean_fitness", "population mean fitness")):
    series = np.array([[float(h[key]) for h in hist if h["seed"] == str(s)] for s in seeds])
    for s, ys in zip(seeds, series):
        ax.plot(ys, lw=0.8, alpha=0.5, label=f"seed {s}")
    median = np.median(series, axis=0)
    ax.plot(median, color="black", lw=2, label="median")
    ax.set_xlabel("generation")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    print(f"  {key}: gens 0-4 {median[:5].mean():.2f}, 15-19 {median[15:20].mean():.2f}, "
          f"30-34 {median[30:35].mean():.2f}, 55-59 {median[-5:].mean():.2f}")
axes[0].legend(fontsize=7)
fig.suptitle("Training on 8 shared random 12x12 mazes per generation")
fig.savefig(f"{OUT}/training_curves.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ---- turning bias and light wiring ----
genomes = np.load(f"{OUT}/evolved_genomes.npz")
visited = np.load(f"{OUT}/visited_inputs.npz")
ga = GeneticAlgorithm(population_size=4, hidden_size=4, seed=0, light_occlusion="blocked")
print("\nTURNING BIAS AND LIGHT WIRING (physical convention: R > L turns LEFT)")
wiring = []
for s in seeds:
    net = ga.decode_genome(genomes[f"seed{s}"])
    states = visited[f"seed{s}"]
    outputs = np.array([net.forward(x) for x in states])
    turn_cmd = outputs[:, 1] - outputs[:, 0]          # R - L: positive turns left
    bias = float(np.median(turn_cmd))
    jac = np.zeros((2, 2))
    for x in states:
        base = net.forward(x)
        for j, idx in enumerate((3, 4)):
            bumped = np.array(x, dtype=float)
            bumped[idx] += 1e-4
            jac[:, j] += (net.forward(bumped) - base) / 1e-4
    jac /= len(states)
    dl_ll, dl_lr = jac[0]
    dr_ll, dr_lr = jac[1]
    same, crossed = abs(dl_ll) + abs(dr_lr), abs(dl_lr) + abs(dr_ll)
    topology = "same-side" if same >= crossed else "crossed"
    sign_sum = (dl_ll + dr_lr) if topology == "same-side" else (dl_lr + dr_ll)
    sign = "excitatory" if sign_sum > 0 else "inhibitory"
    classic = {("same-side", "excitatory"): "fear (2a)", ("crossed", "excitatory"): "aggression (2b)",
               ("same-side", "inhibitory"): "love (3a)", ("crossed", "inhibitory"): "explorer (3b)"}[(topology, sign)]
    # turn toward light on the left: (R - L) should increase with left light more than right light
    turn = (dr_ll - dl_ll) - (dr_lr - dl_lr)
    speed = (dl_ll + dr_ll) + (dl_lr + dr_lr)
    behaves = {(True, True): "aggression-like", (True, False): "love-like",
               (False, True): "fear-like", (False, False): "explorer-like"}[(turn > 0, speed > 0)]
    lit = float(np.mean(states[:, 3:5].max(axis=1) > 0.02))
    wiring.append({"seed": s, "turn_bias_R_minus_L": round(bias, 3),
                   "bias_direction": "left" if bias > 0 else "right",
                   "dL/dlightL": round(dl_ll, 3), "dL/dlightR": round(dl_lr, 3),
                   "dR/dlightL": round(dr_ll, 3), "dR/dlightR": round(dr_lr, 3),
                   "wiring": f"{topology} {sign}", "classic_label": classic,
                   "turns": "toward light" if turn > 0 else "away from light",
                   "turn_strength": round(turn, 3), "speed_in_light": "faster" if speed > 0 else "slower",
                   "behaves_like": behaves, "share_of_states_lit": round(lit, 2),
                   "goals_A": per_seed[s]["course_A"], "goals_B": per_seed[s]["course_B"]})
    print(f"  seed{s}: turning bias {bias:+.2f} ({'left' if bias > 0 else 'right'}); light "
          f"[dL/dlL {dl_ll:+.2f}, dL/dlR {dl_lr:+.2f}; dR/dlL {dr_ll:+.2f}, dR/dlR {dr_lr:+.2f}] "
          f"-> {topology} {sign} = classic {classic}; turns {'TOWARD' if turn > 0 else 'AWAY from'} "
          f"light ({turn:+.2f}), {'faster' if speed > 0 else 'slower'} -> {behaves}; "
          f"A {per_seed[s]['course_A']}/10, B {per_seed[s]['course_B']}/10")
with open(f"{OUT}/turning_bias_and_light_wiring.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(wiring[0]))
    w.writeheader()
    w.writerows(wiring)

# ---- trajectory plots ----
test_mazes = sample_random_mazes(np.random.default_rng(2026), 50)
courses = {name: (maze, start, heading) for name, _, maze, start, heading in make_evaluation_courses()}


def run(net, maze, start, heading, noise_seed):
    vehicle = Vehicle(controller=net, noise_stdev=0.05, noise_seed=noise_seed, navigation_heuristics=False,
                      start_position=start, start_heading=heading, light_occlusion="blocked",
                      **DEFAULT_ROBOT_NOISE)
    world = World(maze=maze, vehicle=vehicle)
    limit = min(10000, course_step_limit(maze, start=start))
    hits = []
    while world.steps < limit and not world.reached_goal:
        c = world.collisions
        world.step()
        if world.collisions > c:
            hits.append(world.position_history[-1])
    return world, np.array(world.position_history), hits, limit


def draw(ax, maze, path, hits, title):
    for (x1, y1), (x2, y2) in maze.walls:
        ax.plot([x1, x2], [y1, y2], color="#333", lw=2)
    ax.add_patch(Circle(maze.goal, maze.goal_radius, color="#e6b800", alpha=0.6))
    sc = ax.scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), cmap="viridis", s=3)
    ax.plot(*path[0], "o", color="#1f5fbf", ms=8)
    if hits:
        h = np.array(hits)
        ax.plot(h[:, 0], h[:, 1], "x", color="#d62728", ms=5)
    ax.set_xlim(maze.x_min, maze.x_max)
    ax.set_ylim(maze.y_min, maze.y_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    return sc


ranked = sorted(seeds, key=lambda s: -(per_seed[s]["random_12x12"] + per_seed[s]["course_A"]
                                         + per_seed[s]["course_B"] + per_seed[s]["course_C"]))
for s in ranked[:2]:
    net = ga.decode_genome(genomes[f"seed{s}"])
    sel = [r for r in rows if r["controller"] == "evolved" and r["seed"] == str(s) and r["set"] == "random_12x12"]
    goals = [int(r["maze"][1:]) for r in sel if float(r["goal_success"])]
    fails = [int(r["maze"][1:]) for r in sel if not float(r["goal_success"])]
    picks = (goals + fails)[:2] if len(goals) >= 2 else (goals + fails)[:2]
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    for ax, idx in zip(axes[:2], picks):
        rm = test_mazes[idx]
        world, path, hits, limit = run(net, rm.maze, rm.start, rm.start_heading, 700000 + idx)
        sc = draw(ax, rm.maze, path, hits, f"random test maze {idx}: {'GOAL' if world.reached_goal else 'no goal'} "
                  f"in {world.steps}/{limit}\n{100 * world.collisions / max(world.steps, 1):.0f}% steps in collision")
    for ax, name in zip(axes[2:], ("course_A", "course_B", "course_C")):
        maze, start, heading = courses[name]
        world, path, hits, limit = run(net, maze, start, heading, 500000)
        sc = draw(ax, maze, path, hits, f"{name} (held out): {'GOAL' if world.reached_goal else 'no goal'} "
                  f"in {world.steps}/{limit}\n{100 * world.collisions / max(world.steps, 1):.0f}% steps in collision")
    fig.colorbar(sc, ax=axes, fraction=0.01, label="step")
    fig.suptitle(f"Evolved controller, seed {s} (blue = start, yellow = goal light, red x = collision)")
    fig.savefig(f"{OUT}/trajectories_seed{s}.png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"plot: trajectories_seed{s}.png (random mazes {picks})")
