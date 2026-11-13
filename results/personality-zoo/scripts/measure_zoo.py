"""Personality fingerprints: every parent and every evolved robot (danger runs + zoo) in the SAME
plain locked test world on 50 mazes (first 50 of the measurement set: seed 4242, noise 900000+).

Also: the fear runs' generation-best robots on 20 of those mazes (clearance over generations),
and one trajectory per world on the same maze.
Usage: python measure_zoo.py REPO GA_NPZ FEAR_NPZ DANGER_DIR ZOO_DIR OUT_DIR [WORKERS]
"""

import csv
import sys
from multiprocessing import Pool

REPO, GA, FEAR, DANGER, ZOO, OUT = sys.argv[1:7]
WORKERS = int(sys.argv[7]) if len(sys.argv) > 7 else 9
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from evolution.metrics import random_maze_episodes, record_run, run_metrics  # noqa: E402

import os  # noqa: E402
EPISODES = random_maze_episodes(int(os.environ.get("ZOO_MAZES", 50)), seed=4242, noise_seed_base=900000)
KEYS = ("speed", "wall_clearance", "pct_steps_in_collision", "exploration", "turning_bias",
        "reverse_fraction", "energy_per_step", "goal_reached")
LABELS = {"speed": "speed", "wall_clearance": "wall clearance", "pct_steps_in_collision": "% steps in collision",
          "exploration": "exploration", "turning_bias": "turning bias (+ left)",
          "reverse_fraction": "reverse fraction", "energy_per_step": "energy per step",
          "goal_reached": "goal success"}
GROUPS = ["safe", "walls hurt", "hurt + hidden", "battery", "explorer", "fear: safe", "fear: walls hurt"]
COLORS = {"seed3": "#eb6834", "seed2": "#2a78d6", "seed19": "#eda100", "seed13": "#1baf7a", "fear": "#e87ba4"}


def decode(genome):
    return GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked").decode_genome(genome)


def measure(args):
    label, genome, count = args
    network = decode(genome)
    runs = [run_metrics(record_run(network, m, s, h, noise_seed=n)) for m, s, h, n in EPISODES[:count]]
    return label, {k: float(np.nanmean([r[k] for r in runs])) for k in KEYS}


def robots():
    ga, fear = np.load(GA), np.load(FEAR)
    danger, zoo = np.load(f"{DANGER}/evolved_genomes.npz"), np.load(f"{ZOO}/evolved_genomes.npz")
    out = [(p, "parent", p, ga[p]) for p in ("seed3", "seed2", "seed19", "seed13")]
    out.append(("fear", "parent", "fear", fear["fear"]))
    for p in ("seed3", "seed19", "seed2", "seed13"):
        out.append((f"{p}_A", "walls hurt", p, danger[f"{p}_A_walls_hurt"]))
        out.append((f"{p}_B", "hurt + hidden", p, danger[f"{p}_B_hurt_hidden_light"]))
    group_of = {"safe": "safe", "battery": "battery", "explorer": "explorer"}
    for tag in zoo.files:
        parent, rest = tag.split("_", 1)
        world = rest.rsplit("_", 1)[0]
        group = (f"fear: {world.replace('_', ' ')}" if parent == "fear" else group_of[world])
        out.append((tag, group, parent, zoo[tag]))
    return out


if __name__ == "__main__":
    specs = robots()
    best_gen = np.load(f"{ZOO}/best_per_generation.npz")
    fear_tags = [t for t in best_gen.files if t.startswith("fear")]
    jobs = [(label, genome, 50) for label, _, _, genome in specs]
    jobs += [(f"{t}@gen{g}", best_gen[t][g], 20) for t in fear_tags for g in range(len(best_gen[t]))]
    with Pool(WORKERS) as pool:
        measured = dict(pool.map(measure, jobs))

    rows = []
    for label, group, parent, _ in specs:
        rows.append({"robot": label, "group": group, "parent": parent, **measured[label]})
    with open(f"{OUT}/fingerprints.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    parent_row = {r["parent"]: r for r in rows if r["group"] == "parent"}
    report = open(f"{OUT}/report.txt", "w")

    def out(text=""):
        print(text)
        print(text, file=report)

    out("ALL ROBOTS IN THE PLAIN TEST WORLD (50 mazes)")
    out(f"{'robot':26} {'group':16} " + " ".join(f"{k[:9]:>9}" for k in KEYS))
    for r in rows:
        out(f"{r['robot']:26} {r['group']:16} " + " ".join(f"{r[k]:9.3f}" for k in KEYS))

    out("\nCHANGE vs PARENT (evolved - parent), mean over robots in each group")
    out(f"{'group':16} {'n':>2} " + " ".join(f"{k[:9]:>9}" for k in KEYS))
    for group in GROUPS:
        members = [r for r in rows if r["group"] == group]
        if not members:
            continue
        deltas = {k: np.mean([r[k] - parent_row[r["parent"]][k] for r in members]) for k in KEYS}
        out(f"{group:16} {len(members):2d} " + " ".join(f"{deltas[k]:+9.3f}" for k in KEYS))
    out("\nchange vs parent minus the SAFE-control change (seeds 3 and 2 only)")
    safe = {r["parent"]: r for r in rows if r["group"] == "safe"}
    for group in GROUPS[1:5]:
        members = [r for r in rows if r["group"] == group and r["parent"] in safe]
        diffs = {k: np.mean([r[k] - safe[r["parent"]][k] for r in members]) for k in KEYS}
        out(f"{group:16} " + " ".join(f"{diffs[k]:+9.3f}" for k in KEYS))
    out("\nturning bias parent -> evolved")
    for r in rows:
        if r["group"] != "parent":
            p = parent_row[r["parent"]]["turning_bias"]
            flag = " FLIPPED" if np.sign(p) != np.sign(r["turning_bias"]) and abs(p) > 0.3 else ""
            out(f"  {r['robot']:26} {p:+.2f} -> {r['turning_bias']:+.2f}{flag}")

    # fingerprint chart: change vs parent per measure, one column per world
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharex=True)
    for ax, key in zip(axes.ravel(), KEYS):
        for x, group in enumerate(GROUPS):
            members = [r for r in rows if r["group"] == group]
            for i, r in enumerate(members):
                offset = (i - (len(members) - 1) / 2) * 0.12
                ax.plot(x + offset, r[key] - parent_row[r["parent"]][key], "o", ms=8,
                        color=COLORS[r["parent"]], mec="white", mew=1.5)
        ax.axhline(0, color="#888", lw=1)
        ax.axvline(4.5, color="#ccc", lw=1, ls="--")
        ax.set_title(f"{LABELS[key]}: evolved - parent", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUPS, rotation=35, ha="right", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=c, mec="white", label=n)
               for n, c in (("parent seed 3 (left-turner)", COLORS["seed3"]), ("parent seed 19 (left)", COLORS["seed19"]),
                            ("parent seed 2 (right-turner)", COLORS["seed2"]), ("parent seed 13 (right)", COLORS["seed13"]),
                            ("parent: hand-set fear robot", COLORS["fear"]))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=5, fontsize=9, frameon=False)
    fig.suptitle("Personality fingerprints: change from parent, all measured in the same plain world (50 mazes). "
                 "0 = same as parent.", y=1.005, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/fingerprint.png", dpi=110, bbox_inches="tight")

    # fear over generations
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ga_clear = [parent_row[p]["wall_clearance"] for p in ("seed3", "seed2", "seed19", "seed13")]
    for ax, key in zip(axes, ("wall_clearance", "goal_reached", "speed")):
        for t, style in zip(sorted(fear_tags), ("-", "--", "-", "--")):
            values = [measured[f"{t}@gen{g}"][key] for g in range(len(best_gen[t]))]
            color = "#2a78d6" if "safe" in t else "#eb6834"
            ax.plot(range(len(values)), values, style, color=color, lw=2,
                    label=("safe world" if "safe" in t else "walls hurt") + f" (run {t[-1]})")
        ax.axhline(parent_row["fear"][key], color=COLORS["fear"], lw=1.5, ls=":", label="fear parent")
        if key == "wall_clearance":
            ax.axhspan(min(ga_clear), max(ga_clear), color="#bbb", alpha=0.4, label="GA winners' range")
        ax.set_xlabel("generation")
        ax.set_title(f"{LABELS[key]} of each generation's best (20 plain mazes)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fear_over_generations.png", dpi=110, bbox_inches="tight")
    out("\nfear runs: clearance gen 0 -> last (20 mazes)")
    for t in sorted(fear_tags):
        last = len(best_gen[t]) - 1
        out(f"  {t}: {measured[f'{t}@gen0']['wall_clearance']:.3f} -> {measured[f'{t}@gen{last}']['wall_clearance']:.3f}; "
            f"goal {measured[f'{t}@gen0']['goal_reached']:.2f} -> {measured[f'{t}@gen{last}']['goal_reached']:.2f}")

    # one trajectory per world, same maze (seed-3 lineage + fear lineage)
    panels = [("seed3", "parent seed 3"), ("seed3_safe_0", "safe"), ("seed3_A", "walls hurt"),
              ("seed3_B", "hurt + hidden"), ("seed3_battery_0", "battery"), ("seed3_explorer_0", "explorer"),
              ("fear", "parent: fear robot"), ("fear_safe_0", "fear: safe"), ("fear_walls_hurt_0", "fear: walls hurt")]
    genomes = {label: genome for label, _, _, genome in specs}
    networks = {label: decode(genomes[label]) for label, _ in panels}
    chosen = 0
    from evolution.experiment import course_step_limit
    for i, (m, s, h, n) in enumerate(EPISODES):
        if course_step_limit(m, start=s) >= 1500 and record_run(networks["seed3"], m, s, h, noise_seed=n).reached_goal and \
                record_run(networks["fear"], m, s, h, noise_seed=n).reached_goal:
            chosen = i
            break
    m, s, h, n = EPISODES[chosen]
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.4))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, title) in zip(list(axes[0]) + list(axes[1][:4]), panels):
        rec = record_run(networks[label], m, s, h, noise_seed=n)
        for (a, b), (c, d) in m.walls:
            ax.plot([a, c], [b, d], color="#333", lw=2)
        ax.add_patch(Circle(m.goal, m.goal_radius, color="#e6b800", alpha=.6))
        ax.add_patch(Circle(m.goal, rec.world.goal_tolerance, fill=False, ls="--", color="#b38f00"))
        ax.plot(rec.positions[:, 0], rec.positions[:, 1], color="#c0392b", lw=1)
        ax.plot(*s, "o", color="#1f5fbf", ms=7)
        ax.set_aspect("equal")
        r = run_metrics(rec)
        ax.set_title(f"{title}\n{'goal in ' + str(rec.steps) + ' steps' if rec.reached_goal else 'no goal'}; "
                     f"clearance {r['wall_clearance']:.2f}, speed {r['speed']:.2f}", fontsize=9)
    fig.suptitle(f"One robot per world on the same plain maze (measurement maze {chosen}); "
                 "top row: seed-3 lineage, bottom row: fear lineage", fontsize=11)
    fig.savefig(f"{OUT}/trajectories.png", dpi=100, bbox_inches="tight")
    out(f"\ntrajectory maze: measurement maze {chosen}")
    report.close()
