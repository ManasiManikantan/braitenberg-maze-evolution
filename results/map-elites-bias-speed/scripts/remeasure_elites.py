"""Re-measure every MAP-Elites elite and make the report figures and tables.

Each elite runs on the 100-maze measurement set (seed 4242, noise seeds 900000+)
and on courses A, B, C x 10 trials (noise seeds 500000+t), using the locked
episode settings of evolution/network_map_elites.py.

Usage: python remeasure_elites.py REPO ME_OUT_DIR GA_MEASURES_CSV GA_GENOMES_NPZ [WORKERS]
  GA_MEASURES_CSV: measure_table.py output (20 GA seeds + both wall followers, new speed metric)
Writes into ME_OUT_DIR: elites_remeasured.csv, who_wins_where.csv, archive_success.png,
trajectories.png, report.txt
"""

import csv
import sys
from multiprocessing import Pool

REPO, ME, GA_MEASURES, GA_GENOMES = sys.argv[1:5]
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 9
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from evolution import network_map_elites as nme  # noqa: E402
from evolution.experiment import make_evaluation_courses  # noqa: E402
from evolution.metrics import random_maze_episodes, record_run  # noqa: E402
from evolution.experiment import course_step_limit  # noqa: E402
from sim.robot import DEFAULT_ROBOT_NOISE  # noqa: E402

EPISODES = random_maze_episodes(100, seed=4242, noise_seed_base=900000)
COURSES = [(name, maze, start, heading) for name, _, maze, start, heading in make_evaluation_courses()]
TRIALS = 10
NEUTRAL = 0.3


def remeasure(genome):
    network = nme.make_ga().decode_genome(genome)
    inputs = []
    runs = [nme.run_episode(network, maze, start, heading, noise_seed, inputs)
            for maze, start, heading, noise_seed in EPISODES]
    row = nme.summarize_runs(network, runs, inputs)
    for name, maze, start, heading in COURSES:
        row[name] = float(np.mean([
            nme.run_episode(network, maze, start, heading, 500000 + t)[1]["goal_reached"]
            for t in range(TRIALS)]))
    return row


def speed_class(speed, low, high):
    third = (high - low) / 3
    return "slow" if speed < low + third else ("fast" if speed >= high - third else "medium")


def bias_class(bias):
    return "right" if bias < -NEUTRAL else ("left" if bias > NEUTRAL else "neutral")


if __name__ == "__main__":
    data = np.load(f"{ME}/archive.npz")
    cells, genomes = data["cells"], data["genomes"]
    low, high = (float(v) for v in data["speed_range"])
    bins = tuple(int(b) for b in data["bins"])
    import os
    if os.environ.get("PLOT_ONLY"):
        old = list(csv.DictReader(open(f"{ME}/elites_remeasured.csv")))
        measured = [{"goal_rate": float(r["random_success"]), "fitness": float(r["mean_fitness_100"]),
                     **{k: float(r[k]) for k in ("turning_bias", "speed", "reverse_fraction",
                        "pct_steps_in_collision", "exploration", "wall_side", "light_reliance",
                        "course_A", "course_B", "course_C")}} for r in old]
    else:
        with Pool(WORKERS) as pool:
            measured = pool.map(remeasure, list(genomes))

    rows = []
    for index, (cell, row) in enumerate(zip(cells, measured)):
        rows.append({
            "elite": index, "cell_bias": int(cell[0]), "cell_speed": int(cell[1]),
            "search_fitness": float(data["fitness"][index]),
            "search_turning_bias": float(data["turning_bias"][index]),
            "search_speed": float(data["speed"][index]),
            "random_success": row["goal_rate"], "turning_bias": row["turning_bias"], "speed": row["speed"],
            "reverse_fraction": row["reverse_fraction"],
            "pct_steps_in_collision": row["pct_steps_in_collision"],
            "exploration": row["exploration"], "wall_side": row["wall_side"],
            "light_reliance": row["light_reliance"], "mean_fitness_100": row["fitness"],
            "course_A": row["course_A"], "course_B": row["course_B"], "course_C": row["course_C"],
            "bias_class": bias_class(row["turning_bias"]),
            "speed_class": speed_class(row["speed"], low, high),
        })
    with open(f"{ME}/elites_remeasured.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    report = open(f"{ME}/report.txt", "w")

    def out(text=""):
        print(text)
        print(text, file=report)

    success = np.array([r["random_success"] for r in rows])
    out(f"filled cells: {len(rows)}/{bins[0] * bins[1]}")
    out(f"elites with random-maze success >= 0.70: {int(np.sum(success >= 0.7))}, >= 0.50: "
        f"{int(np.sum(success >= 0.5))}; best {success.max():.2f}")

    # who wins where (100-maze bias and speed)
    table = []
    out("\nWHO WINS WHERE (elites grouped by re-measured bias x speed; mean success)")
    out(f"{'bias':8} {'speed':7} {'n':>3} {'random':>7} {'best':>5} {'A':>5} {'B':>5} {'C':>5}")
    for b in ("right", "neutral", "left"):
        for s in ("slow", "medium", "fast"):
            group = [r for r in rows if r["bias_class"] == b and r["speed_class"] == s]
            entry = {"bias": b, "speed": s, "n": len(group)}
            if group:
                entry.update({k: float(np.mean([r[k] for r in group]))
                              for k in ("random_success", "course_A", "course_B", "course_C")})
                entry["best_random"] = float(max(r["random_success"] for r in group))
                out(f"{b:8} {s:7} {len(group):3d} {entry['random_success']:7.2f} {entry['best_random']:5.2f} "
                    f"{entry['course_A']:5.2f} {entry['course_B']:5.2f} {entry['course_C']:5.2f}")
            else:
                out(f"{b:8} {s:7}   0")
            table.append(entry)
    with open(f"{ME}/who_wins_where.csv", "w", newline="") as f:
        fields = ["bias", "speed", "n", "random_success", "best_random", "course_A", "course_B", "course_C"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    speeds = np.array([r["speed"] for r in rows])
    rx, ry = np.argsort(np.argsort(speeds)), np.argsort(np.argsort(success))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    out(f"\nspeed vs random success among elites: Spearman rho {rho:+.2f}")
    for s in ("slow", "medium", "fast"):
        group = [r["random_success"] for r in rows if r["speed_class"] == s]
        if group:
            out(f"  {s}: {len(group)} elites, best {max(group):.2f}, "
                f"number >= 0.70: {sum(g >= 0.7 for g in group)}")
    out("\nper-elite logged stats (100 mazes): mean (min-max)")
    for k in ("reverse_fraction", "pct_steps_in_collision", "exploration", "light_reliance"):
        v = np.array([r[k] for r in rows])
        out(f"  {k}: {v.mean():.3f} ({v.min():.3f}-{v.max():.3f})")
    rel = np.array([r["light_reliance"] for r in rows])
    out(f"  light reliance vs random success: Spearman rho "
        f"{np.corrcoef(np.argsort(np.argsort(rel)), ry)[0, 1]:+.2f}")

    # heatmap colored by random-maze success
    grid = np.full((bins[1], bins[0]), np.nan)
    for r in rows:
        grid[r["cell_speed"], r["cell_bias"]] = r["random_success"]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e5e5")
    image = ax.imshow(grid, origin="lower", extent=(-1, 1, low, high), aspect="auto", cmap=cmap,
                      vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="goal success on 100 random 12x12 mazes")
    for r in rows:
        cx = -1 + (r["cell_bias"] + 0.5) * 2 / bins[0]
        cy = low + (r["cell_speed"] + 0.5) * (high - low) / bins[1]
        ax.text(cx, cy, f"{r['random_success']:.2f}", ha="center", va="center", fontsize=6,
                color="white" if r["random_success"] < 0.6 else "black")
    ga_rows = list(csv.DictReader(open(GA_MEASURES)))
    clip = lambda v: min(max(v, low + 0.003), high - 0.003)  # noqa: E731
    first = True
    for g in ga_rows:
        bias, speed = float(g["turning_bias_mean"]), float(g["speed_mean"])
        if g["controller"].startswith("run"):
            ax.plot(bias, clip(speed), "o", ms=7, mfc="none", mec="red", mew=1.5,
                    label="GA seeds (20)" if first else None)
            ax.annotate(g["controller"].split("seed")[1], (bias, clip(speed)), fontsize=6, color="red",
                        xytext=(3, 3), textcoords="offset points")
            first = False
        else:
            marker = "s" if g["controller"].startswith("left") else "D"
            ax.plot(bias, clip(speed), marker, ms=9, mfc="white", mec="black", mew=1.5,
                    label=g["controller"])
    ax.set_xlabel("turning bias (+ = turns left)")
    ax.set_ylabel(f"speed (mean |forward speed| / max); edge rows include values beyond {low}-{high}")
    ax.set_title(f"MAP-Elites archive: {len(rows)}/{bins[0] * bins[1]} cells, colored by random-maze "
                 "success\n(cells placed by search descriptors; markers at 100-maze measurements)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.savefig(f"{ME}/archive_success.png", dpi=120, bbox_inches="tight")

    # three very different good elites on the same maze
    def best_of(condition):
        group = [r for r in rows if condition(r)] or rows
        return max(group, key=lambda r: r["random_success"])

    picks = [best_of(lambda r: r["bias_class"] == "left" and r["speed_class"] == "fast"),
             best_of(lambda r: r["bias_class"] == "neutral"),
             best_of(lambda r: r["bias_class"] == "right" and r["speed_class"] == "slow")]
    networks = [nme.make_ga().decode_genome(genomes[p["elite"]]) for p in picks]
    maze_index = None
    for i, (maze, start, heading, noise_seed) in enumerate(EPISODES):
        if course_step_limit(maze, start=start) < 1500:
            continue
        reached = [record_run(n, maze, start, heading, noise_seed=noise_seed, robot_noise=DEFAULT_ROBOT_NOISE,
                              light_occlusion="blocked").reached_goal for n in networks]
        if all(reached):
            maze_index = i
            break
    maze_index = 0 if maze_index is None else maze_index
    maze, start, heading, noise_seed = EPISODES[maze_index]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.6))
    for ax, p, n in zip(np.atleast_1d(axes), picks, networks):
        rec = record_run(n, maze, start, heading, noise_seed=noise_seed, robot_noise=DEFAULT_ROBOT_NOISE,
                         light_occlusion="blocked")
        pos = rec.positions
        for (a, b), (c, d) in maze.walls:
            ax.plot([a, c], [b, d], color="#333", lw=2)
        ax.add_patch(Circle(maze.goal, maze.goal_radius, color="#e6b800", alpha=.6))
        ax.add_patch(Circle(maze.goal, rec.world.goal_tolerance, fill=False, ls="--", color="#b38f00"))
        ax.plot(pos[:, 0], pos[:, 1], color="#c0392b", lw=1)
        ax.plot(*start, "o", color="#1f5fbf", ms=7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"elite {p['elite']}: bias {p['turning_bias']:+.2f}, speed {p['speed']:.2f}, "
                     f"reverse {p['reverse_fraction']:.2f}\nrandom {p['random_success']:.2f}, "
                     f"A/B/C {p['course_A']:.1f}/{p['course_B']:.1f}/{p['course_C']:.1f}; this maze: "
                     f"{'goal in ' + str(rec.steps) + ' steps' if rec.reached_goal else 'no goal'}",
                     fontsize=8)
    fig.suptitle(f"Three different elites on measurement maze {maze_index}", fontsize=10)
    fig.savefig(f"{ME}/trajectories.png", dpi=110, bbox_inches="tight")
    out(f"\ntrajectory elites: {[p['elite'] for p in picks]} on measurement maze {maze_index}")
    report.close()
