"""Before/after measures for the dangerous-world runs.

Every original GA winner and every evolved robot runs on the same 50 random
12x12 mazes (first 50 of the 100-maze measurement set: seed 4242, noise seeds
900000+) in all three worlds (baseline, A walls hurt, B walls hurt + hidden light).
Usage: python measure_danger.py REPO GA_GENOMES_NPZ DANGER_DIR [WORKERS]
Writes DANGER_DIR/measures.csv, before_after.csv, report.txt
"""

import csv
import sys
from multiprocessing import Pool

REPO, GA_GENOMES, OUT = sys.argv[1:4]
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 9
sys.path.insert(0, REPO)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

from evolution.genetic_algorithm import GeneticAlgorithm  # noqa: E402
from evolution.metrics import random_maze_episodes, record_run, run_metrics  # noqa: E402
from evolution.experiment import course_step_limit  # noqa: E402
from evolve_danger import NOISE, ROBOTS, STEP_CAP, WORLDS, world_settings  # noqa: E402

import os  # noqa: E402
EPISODES = random_maze_episodes(int(os.environ.get("DANGER_MAZES", 50)), seed=4242, noise_seed_base=900000)
KEYS = ("speed", "wall_clearance", "pct_steps_in_collision", "exploration", "turning_bias",
        "reverse_fraction", "goal_reached")


def decode(genome):
    return GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked").decode_genome(genome)


def job(spec):
    label, robot, trained_in, genome, world_name = spec
    network = decode(genome)
    world_options, robot_noise = world_settings(world_name)
    runs = []
    for maze, start, heading, noise_seed in EPISODES:
        record = record_run(network, maze, start, heading,
                            max_steps=min(STEP_CAP, course_step_limit(maze, start=start)),
                            noise_seed=noise_seed, noise_stdev=NOISE, robot_noise=robot_noise,
                            light_occlusion="blocked", world_options=world_options)
        m = run_metrics(record)
        m["hits"] = record.world.contact_entries
        m["died"] = float(record.world.dead)
        runs.append(m)
    row = {"controller": label, "robot": robot, "trained_in": trained_in, "measured_in": world_name}
    for k in KEYS + ("hits", "died"):
        row[k] = float(np.mean([r[k] for r in runs]))
        row[k + "_std"] = float(np.std([r[k] for r in runs]))
    return row


if __name__ == "__main__":
    ga = np.load(GA_GENOMES)
    evolved = np.load(f"{OUT}/evolved_genomes.npz")
    specs = []
    for robot in ROBOTS:
        sources = [("original", "GA baseline", ga[f"seed{robot}"])]
        sources += [(f"evolved in {w}", w, evolved[f"seed{robot}_{w}"])
                    for w in ("A_walls_hurt", "B_hurt_hidden_light")]
        for label, trained_in, genome in sources:
            specs += [(label, robot, trained_in, genome, w) for w in WORLDS]
    with Pool(WORKERS) as pool:
        rows = pool.map(job, specs)
    with open(f"{OUT}/measures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def get(robot, trained_in, measured_in):
        return next(r for r in rows if r["robot"] == robot and r["trained_in"] == trained_in
                    and r["measured_in"] == measured_in)

    report = open(f"{OUT}/report.txt", "w")

    def out(text=""):
        print(text)
        print(text, file=report)

    table = []
    out("BEFORE (original GA winner) vs AFTER (evolved), behavior measured in the robot's new world; "
        "50 mazes")
    header = (f"{'robot':>5} {'world':6} {'version':8} {'speed':>6} {'clear':>6} {'coll%':>6} {'hits':>5} "
              f"{'died':>5} {'explore':>7} {'bias':>6} {'goal new':>8} {'goal base':>9}")
    for world, short in (("A_walls_hurt", "A"), ("B_hurt_hidden_light", "B")):
        out()
        out(header)
        for robot in ROBOTS:
            for version, trained_in in (("before", "GA baseline"), ("after", world)):
                r = get(robot, trained_in, world)
                base = get(robot, trained_in, "baseline")
                entry = {"robot": robot, "world": short, "version": version,
                         **{k: round(r[k], 3) for k in KEYS + ("hits", "died")},
                         "goal_new_world": round(r["goal_reached"], 3),
                         "goal_baseline": round(base["goal_reached"], 3)}
                table.append(entry)
                out(f"{robot:5d} {short:6} {version:8} {r['speed']:6.2f} {r['wall_clearance']:6.3f} "
                    f"{r['pct_steps_in_collision']:6.1f} {r['hits']:5.1f} {r['died']:5.2f} "
                    f"{r['exploration']:7.2f} {r['turning_bias']:+6.2f} {r['goal_reached']:8.2f} "
                    f"{base['goal_reached']:9.2f}")
    with open(f"{OUT}/before_after.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    out("\nCHANGE after - before (in new world), per world: mean over the 4 robots [count of robots that moved "
        "in the 'careful' direction]")
    for world, short in (("A_walls_hurt", "A"), ("B_hurt_hidden_light", "B")):
        parts = []
        for k, careful in (("wall_clearance", +1), ("speed", -1), ("pct_steps_in_collision", -1),
                           ("hits", -1), ("died", -1), ("exploration", +1), ("goal_reached", +1)):
            d = [get(rb, world, world)[k] - get(rb, "GA baseline", world)[k] for rb in ROBOTS]
            parts.append(f"{k} {np.mean(d):+.3f} [{sum(np.sign(x) == careful for x in d)}/4]")
        out(f"  {short}: " + "; ".join(parts))
    out("\nturning bias before -> after (measured in baseline world):")
    for robot in ROBOTS:
        out(f"  seed {robot}: {get(robot, 'GA baseline', 'baseline')['turning_bias']:+.2f} -> "
            f"A {get(robot, 'A_walls_hurt', 'baseline')['turning_bias']:+.2f}, "
            f"B {get(robot, 'B_hurt_hidden_light', 'baseline')['turning_bias']:+.2f}")
    out("\nexploration, B-evolved vs A-evolved (both measured in world B):")
    for robot in ROBOTS:
        out(f"  seed {robot}: A-evolved {get(robot, 'A_walls_hurt', 'B_hurt_hidden_light')['exploration']:.3f}, "
            f"B-evolved {get(robot, 'B_hurt_hidden_light', 'B_hurt_hidden_light')['exploration']:.3f}, "
            f"original {get(robot, 'GA baseline', 'B_hurt_hidden_light')['exploration']:.3f}")
    report.close()
