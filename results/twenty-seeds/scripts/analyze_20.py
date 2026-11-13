"""20-seed analysis: behavior measures, goal success, bias vs A-B preference, turner classes.

Usage: python analyze_20.py OUT_DIR MEASURES_CSV WALL_FOLLOWER_CSV
  OUT_DIR       folder of after_simple.py (test_episodes.csv, history.csv)
  MEASURES_CSV  output of measure_table.py for these seeds + wall followers
  WALL_FOLLOWER_CSV  wf_simple.py output (wall followers on A/B/C under the 8x limit)
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT, MEASURES, WF = sys.argv[1:4]
NEUTRAL = 0.3
rows = list(csv.DictReader(open(f"{OUT}/test_episodes.csv")))
measures = {r["controller"]: r for r in csv.DictReader(open(MEASURES))}
seeds = sorted({int(r["seed"]) for r in rows if r["controller"] == "evolved"})

per_seed = []
for s in seeds:
    sel = [r for r in rows if r["controller"] == "evolved" and r["seed"] == str(s)]
    goal = {name: float(np.mean([float(r["goal_success"]) for r in sel if r["set"] == name]))
            for name in ("random_12x12", "course_A", "course_B", "course_C")}
    m = measures[f"run seed{s}"]
    bias = float(m["turning_bias_mean"])
    cls = "right" if bias < -NEUTRAL else ("left" if bias > NEUTRAL else "neutral")
    per_seed.append({
        "seed": s, "class": cls,
        **{f"{k}": float(m[f"{k}_mean"]) for k in ("turning_bias", "speed", "wall_clearance", "exploration", "wall_side")},
        **{f"{k}_std": float(m[f"{k}_std"]) for k in ("turning_bias", "speed", "wall_clearance", "exploration", "wall_side")},
        "goal_random": goal["random_12x12"], "goal_A": goal["course_A"], "goal_B": goal["course_B"],
        "goal_C": goal["course_C"], "A_minus_B": goal["course_A"] - goal["course_B"],
        "pct_collision_100": float(m["pct_steps_in_collision_mean"]),
    })
with open(f"{OUT}/per_seed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_seed[0]))
    w.writeheader()
    w.writerows(per_seed)

print("PER SEED (behavior on 100 mazes: mean +/- std; goal success on 50 random test mazes, A/B/C x 10)")
print(f"{'seed':>4} {'class':8} {'turn bias':>12} {'speed':>11} {'clearance':>11} {'explore':>11} "
      f"{'wall side':>11} {'random':>7} {'A':>4} {'B':>4} {'C':>4}")
for p in per_seed:
    print(f"{p['seed']:>4} {p['class']:8} "
          + " ".join(f"{p[k]:+.2f}±{p[k + '_std']:.2f}".rjust(11 if k != 'turning_bias' else 12)
                     for k in ("turning_bias", "speed", "wall_clearance", "exploration", "wall_side"))
          + f" {p['goal_random']:7.2f} {p['goal_A']:4.1f} {p['goal_B']:4.1f} {p['goal_C']:4.1f}")
for name in ("left wall follower", "right wall follower"):
    m = measures[name]
    print(f"{name:13} " + " ".join(
        f"{float(m[k + '_mean']):+.2f}±{float(m[k + '_std']):.2f}"
        for k in ("turning_bias", "speed", "wall_clearance", "exploration", "wall_side")))

classes = {c: [p for p in per_seed if p["class"] == c] for c in ("right", "neutral", "left")}
print("\nCLASSES (|bias| < %.1f = neutral)" % NEUTRAL)
for c, members in classes.items():
    if not members:
        print(f"  {c}: 0 seeds")
        continue
    print(f"  {c}: {len(members)} seeds {[p['seed'] for p in members]}; mean goal random "
          f"{np.mean([p['goal_random'] for p in members]):.2f}, A {np.mean([p['goal_A'] for p in members]):.2f}, "
          f"B {np.mean([p['goal_B'] for p in members]):.2f}, C {np.mean([p['goal_C'] for p in members]):.2f}, "
          f"A-B {np.mean([p['A_minus_B'] for p in members]):+.2f}")


def permutation_p(a, b, statistic=np.mean, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    observed = statistic(a) - statistic(b)
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        diff = statistic(pooled[:len(a)]) - statistic(pooled[len(a):])
        count += abs(diff) >= abs(observed) - 1e-12
    return observed, count / n


print("\nLEFT vs RIGHT TURNERS (two-sided permutation test on per-seed values)")
if classes["left"] and classes["right"]:
    for key, label in (("goal_random", "random-maze success"),
                       ("goal_C", "course C success"),
                       ("pct_collision_100", "% steps in collision")):
        a = np.array([p[key] for p in classes["left"]])
        b = np.array([p[key] for p in classes["right"]])
        diff, pval = permutation_p(a, b)
        print(f"  {label}: left {a.mean():.3f} vs right {b.mean():.3f}, difference {diff:+.3f}, p = {pval:.3f}")
    favored_left = np.array([p["goal_A"] for p in classes["left"]])     # A favors left-hand behavior?
    favored_right = np.array([p["goal_B"] for p in classes["right"]])
    print(f"  success on each class's 'own' course (left-turners on A, right-turners on B): "
          f"{favored_left.mean():.2f} vs {favored_right.mean():.2f}")
    a = np.array([max(p["goal_A"], p["goal_B"]) for p in classes["left"]])
    b = np.array([max(p["goal_A"], p["goal_B"]) for p in classes["right"]])
    diff, pval = permutation_p(a, b)
    print(f"  better of A/B: left {a.mean():.2f} vs right {b.mean():.2f}, difference {diff:+.2f}, p = {pval:.3f}")

bias = np.array([p["turning_bias"] for p in per_seed])
amb = np.array([p["A_minus_B"] for p in per_seed])


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


rho = spearman(bias, amb)
rng = np.random.default_rng(1)
null = [spearman(bias, rng.permutation(amb)) for _ in range(20000)]
p_rho = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
print(f"\nturning bias vs (A - B): Spearman rho = {rho:+.2f} (permutation p = {p_rho:.3f})")

fig, ax = plt.subplots(figsize=(7, 5))
colors = {"right": "#2b6cb0", "neutral": "#718096", "left": "#c05621"}
for p in per_seed:
    ax.scatter(p["turning_bias"], p["A_minus_B"], s=40 + 160 * p["goal_random"],
               color=colors[p["class"]], alpha=0.8, edgecolor="black")
    ax.annotate(str(p["seed"]), (p["turning_bias"], p["A_minus_B"]), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
wf_rows = list(csv.DictReader(open(WF)))
for name, key, marker in (("left wall follower", "left_wall", "s"), ("right wall follower", "right_wall", "D")):
    m = measures[name]
    a = np.mean([float(r["goal_within_limit"]) for r in wf_rows if r["controller"] == key and r["set"] == "course_A"])
    b = np.mean([float(r["goal_within_limit"]) for r in wf_rows if r["controller"] == key and r["set"] == "course_B"])
    ax.scatter(float(m["turning_bias_mean"]), a - b, marker=marker, s=90, color="white", edgecolor="black",
               label=f"{name}")
ax.legend(fontsize=8, loc="upper left")
ax.axhline(0, color="#999", lw=1)
for x in (-NEUTRAL, NEUTRAL):
    ax.axvline(x, color="#bbb", lw=1, ls="--")
ax.set_xlabel("turning bias (100 random mazes; + = turns left)")
ax.set_ylabel("course A success - course B success (10 trials each)")
ax.set_title(f"20 evolved seeds: turning bias vs A/B preference (Spearman rho {rho:+.2f}, p {p_rho:.3f})\n"
             "dot size = random-maze success; colors: right / neutral / left turners", fontsize=9)
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.1, 1.1)
ax.grid(alpha=0.3)
fig.savefig(f"{OUT}/bias_vs_AB.png", dpi=110, bbox_inches="tight")
print("plot: bias_vs_AB.png")
