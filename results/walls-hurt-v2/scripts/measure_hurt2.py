"""Walls-hurt v2 vs the zoo's safe control (and the loophole version v1).

Every robot is measured in the SAME plain world on 50 mazes (seed 4242, noise 900000+), and
also in the v2 walls-hurt world (deaths). Fear lineages: generation-best clearance over
generations on 20 plain mazes, v2 vs safe.
Usage: python measure_hurt2.py REPO GA_NPZ FEAR_NPZ DANGER_DIR ZOO_DIR HURT2_DIR [WORKERS]
"""

import csv
import sys
from multiprocessing import Pool

REPO, GA, FEAR, DANGER, ZOO, HURT2 = sys.argv[1:7]
WORKERS = int(sys.argv[7]) if len(sys.argv) > 7 else 9
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from evolution.metrics import random_maze_episodes, record_run, run_metrics  # noqa: E402

EPISODES = random_maze_episodes(50, seed=4242, noise_seed_base=900000)
V2 = {"health": 15, "contact_step_cost": 0.1}
KEYS = ("wall_clearance", "speed", "pct_steps_in_collision", "turning_bias", "exploration",
        "reverse_fraction", "energy_per_step", "goal_reached")
GROUPS = ("parent", "safe control", "walls hurt v1 (loophole)", "walls hurt v2")


def decode(genome):
    return GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked").decode_genome(genome)


def measure(args):
    label, genome, count, world_options = args
    network = decode(genome)
    runs, deaths = [], []
    for m, s, h, n in EPISODES[:count]:
        rec = record_run(network, m, s, h, noise_seed=n, world_options=world_options)
        runs.append(run_metrics(rec))
        deaths.append(float(rec.world.dead))
    out = {k: float(np.nanmean([r[k] for r in runs])) for k in KEYS}
    out["died"] = float(np.mean(deaths))
    return label, out


if __name__ == "__main__":
    ga, fear = np.load(GA), np.load(FEAR)
    danger, zoo = np.load(f"{DANGER}/evolved_genomes.npz"), np.load(f"{ZOO}/evolved_genomes.npz")
    hurt2 = np.load(f"{HURT2}/evolved_genomes.npz")
    robots = [("seed3", "parent", "seed3", ga["seed3"]), ("seed2", "parent", "seed2", ga["seed2"]),
              ("fear", "parent", "fear", fear["fear"])]
    for tag in ("seed3_safe_0", "seed2_safe_0", "fear_safe_0", "fear_safe_1"):
        robots.append((tag, "safe control", tag.split("_")[0], zoo[tag]))
    robots += [("seed3_A (30 gen)", "walls hurt v1 (loophole)", "seed3", danger["seed3_A_walls_hurt"]),
               ("seed2_A (30 gen)", "walls hurt v1 (loophole)", "seed2", danger["seed2_A_walls_hurt"])]
    for tag in ("fear_walls_hurt_0", "fear_walls_hurt_1"):
        robots.append((tag, "walls hurt v1 (loophole)", "fear", zoo[tag]))
    for tag in hurt2.files:
        robots.append((tag, "walls hurt v2", tag.split("_")[0], hurt2[tag]))

    gens_safe = np.load(f"{ZOO}/best_per_generation.npz")
    gens_v2 = np.load(f"{HURT2}/best_per_generation.npz")
    lineages = [(f"safe {t[-1]}", gens_safe[t]) for t in ("fear_safe_0", "fear_safe_1")]
    lineages += [(f"v2 {t[-1]}", gens_v2[t]) for t in sorted(gens_v2.files) if t.startswith("fear")]

    jobs = [(label, genome, 50, None) for label, _, _, genome in robots]
    jobs += [(f"{label}@v2", genome, 50, V2) for label, _, _, genome in robots]
    jobs += [(f"{name}@gen{g}", genomes[g], 20, None) for name, genomes in lineages for g in range(len(genomes))]
    with Pool(WORKERS) as pool:
        measured = dict(pool.map(measure, jobs))

    rows = []
    for label, group, parent, _ in robots:
        rows.append({"robot": label, "group": group, "parent": parent, **measured[label],
                     "goal_in_v2_world": measured[f"{label}@v2"]["goal_reached"],
                     "died_in_v2_world": measured[f"{label}@v2"]["died"]})
    with open(f"{HURT2}/comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = open(f"{HURT2}/report.txt", "w")

    def out(text=""):
        print(text)
        print(text, file=report)

    out("PLAIN-WORLD BEHAVIOR (50 mazes); last two columns: in the v2 walls-hurt world")
    out(f"{'robot':22} {'group':26} {'clear':>6} {'speed':>6} {'touch%':>7} {'bias':>6} {'explore':>7} "
        f"{'goal':>5} | {'goal v2':>7} {'died v2':>7}")
    for r in rows:
        out(f"{r['robot']:22} {r['group']:26} {r['wall_clearance']:6.3f} {r['speed']:6.2f} "
            f"{r['pct_steps_in_collision']:7.1f} {r['turning_bias']:+6.2f} {r['exploration']:7.2f} "
            f"{r['goal_reached']:5.2f} | {r['goal_in_v2_world']:7.2f} {r['died_in_v2_world']:7.2f}")
    out("\nv2 minus safe control, per lineage (plain world)")
    for parent in ("seed3", "seed2", "fear"):
        safe = [r for r in rows if r["group"] == "safe control" and r["parent"] == parent]
        v2 = [r for r in rows if r["group"] == "walls hurt v2" and r["parent"] == parent]
        if safe and v2:
            out(f"  {parent:6} " + "; ".join(
                f"{k} {np.mean([r[k] for r in v2]) - np.mean([r[k] for r in safe]):+.3f}"
                for k in ("wall_clearance", "speed", "pct_steps_in_collision", "goal_reached")))

    # chart: three careful-ness measures per lineage, groups side by side
    colors = {"parent": "#9b9b9b", "safe control": "#2a78d6", "walls hurt v1 (loophole)": "#eda100",
              "walls hurt v2": "#eb6834"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, key, title in zip(axes, ("wall_clearance", "speed", "pct_steps_in_collision", "died_in_v2_world"),
                              ("wall clearance (plain world)", "speed (plain world)",
                               "% steps touching a wall (plain world)", "deaths in the v2 walls-hurt world")):
        for x, parent in enumerate(("seed3", "seed2", "fear")):
            for j, group in enumerate(GROUPS):
                members = [r for r in rows if r["group"] == group and r["parent"] == parent]
                for i, r in enumerate(members):
                    ax.plot(x + (j - 1.5) * 0.2 + (i - (len(members) - 1) / 2) * 0.05, r[key], "o", ms=8,
                            color=colors[group], mec="white", mew=1.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["seed 3 lineage", "seed 2 lineage", "fear lineage"])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=c, mec="white", label=g) for g, c in colors.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{HURT2}/careful_comparison.png", dpi=110, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for name, genomes in lineages:
        values = [measured[f"{name}@gen{g}"]["wall_clearance"] for g in range(len(genomes))]
        ax.plot(range(len(values)), values, "-" if name.endswith("0") else "--", lw=2,
                color="#2a78d6" if name.startswith("safe") else "#eb6834",
                label=("safe world" if name.startswith("safe") else "walls hurt v2") + f", run {name[-1]}")
    ax.axhline(measured["fear"]["wall_clearance"], color="#e87ba4", ls=":", lw=1.5, label="fear parent")
    ga_c = [measured["seed3"]["wall_clearance"], measured["seed2"]["wall_clearance"]]
    ax.axhspan(min(ga_c), max(ga_c), color="#bbb", alpha=0.4, label="GA winners 3 and 2")
    ax.set_xlabel("generation")
    ax.set_ylabel("wall clearance of generation best (20 plain mazes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(f"{HURT2}/fear_clearance_v2_vs_safe.png", dpi=110, bbox_inches="tight")
    out("\nfear lineages, clearance gen 0 -> last:")
    for name, genomes in lineages:
        out(f"  {name}: {measured[f'{name}@gen0']['wall_clearance']:.3f} -> "
            f"{measured[f'{name}@gen{len(genomes) - 1}']['wall_clearance']:.3f}")
    report.close()
