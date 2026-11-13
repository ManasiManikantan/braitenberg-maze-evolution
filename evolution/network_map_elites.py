"""MAP-Elites over network-only controllers in the locked simple-12x12 setup.

Separate from ``evolution/map_elites.py`` (the older course-based search,
left unchanged). Everything except the search itself matches the locked GA:
5-input network (distances at 0/+-60 deg, 2 wall-blocked light sensors),
hidden size 4, outcome-only fitness, default robot noise with a random-side
motor mismatch, heading noise 0.05, random 12x12 perfect mazes with the 8x
shortest-route time limit (capped at 10000 steps), and mutation 0.15 / 0.2
with crossover 0.8.

Archive: 10 x 10 grid over turning bias (-1..1) x speed (``speed_range``).
Each batch picks parents, makes ``batch_size`` children by crossover and
mutation, and evaluates them in parallel on the *same* 8 random mazes and
noise seed, so every controller in a batch is scored on the same mazes.
Fitness = mean outcome fitness over those mazes; the descriptors are the
mean turning bias and mean speed (``evolution/metrics.py``) over the same runs.

Parent selection (instead of uniform random): an elite's weight is
``(1 + empty neighbouring cells) / (1 + times its cell was improved)``, so
elites at the edge of the filled region and cells that rarely improve are
picked more often.

Every elite also logs reverse fraction, % steps in collision, exploration,
wall side, goal rate and light reliance: mean |change in the two motor
outputs| when the light inputs on the visited states are set to 0 (no
light seen).

The search writes a checkpoint (``checkpoint.pkl`` plus ``archive.npz``)
every ``checkpoint_every`` evaluations and can be resumed from it.
"""

import argparse
import csv
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from evolution.experiment import course_step_limit, sample_random_mazes
from evolution.fitness import evaluate_outcome_fitness
from evolution.genetic_algorithm import GeneticAlgorithm
from evolution.metrics import record_run, run_metrics
from sim.robot import DEFAULT_ROBOT_NOISE

LIGHT = "blocked"
NOISE_STDEV = 0.05
STEP_CAP = 10000
MAZES_PER_BATCH = 8
DEFAULT_BINS = (10, 10)
BIAS_RANGE = (-1.0, 1.0)
# 5th-95th percentile of the 20 GA winners' speeds (0.40-0.58 on the
# 100-maze set), widened by 25% of that width on each side.
DEFAULT_SPEED_RANGE = (0.35, 0.63)
LIGHT_SAMPLES = 400
LOGGED_STATS = ("goal_rate", "reverse_fraction", "pct_steps_in_collision", "exploration",
                "wall_side", "light_reliance")


def make_ga(seed=0):
    """GA object used only for genome decoding, crossover and mutation (locked settings)."""
    return GeneticAlgorithm(population_size=2, elite_count=0, tournament_size=1, hidden_size=4,
                            mutation_rate=0.15, mutation_scale=0.2, seed=seed,
                            network_only=True, light_occlusion=LIGHT)


def light_reliance(network, inputs, samples=LIGHT_SAMPLES):
    """Mean |output change| over both motors when the light inputs are set to 0."""
    inputs = np.asarray(inputs, dtype=float)
    if len(inputs) == 0:
        return 0.0
    if len(inputs) > samples:
        inputs = inputs[np.linspace(0, len(inputs) - 1, samples).astype(int)]
    dark = inputs.copy()
    dark[:, 3:] = 0.0
    with_light = np.array([network.forward(x) for x in inputs])
    without = np.array([network.forward(x) for x in dark])
    return float(np.mean(np.abs(with_light - without)))


def run_episode(network, maze, start, heading, noise_seed, inputs=None):
    """One locked-setup run; returns (outcome fitness dict, run_metrics dict)."""
    hook = None if inputs is None else (lambda vehicle: inputs.append(vehicle.controller_inputs()))
    record = record_run(network, maze, start, heading,
                        max_steps=min(STEP_CAP, course_step_limit(maze, start=start)),
                        noise_seed=noise_seed, noise_stdev=NOISE_STDEV,
                        robot_noise=DEFAULT_ROBOT_NOISE, light_occlusion=LIGHT, on_step=hook)
    return evaluate_outcome_fitness(record.world), run_metrics(record)


def summarize_runs(network, runs, inputs):
    """Fitness, descriptors and logged stats from a list of (fitness, metrics) runs."""
    metrics = [m for _, m in runs]

    def mean(name):
        values = np.array([m[name] for m in metrics], dtype=float)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    return {
        "fitness": float(np.mean([f["score"] for f, _ in runs])),
        "turning_bias": mean("turning_bias"),
        "speed": mean("speed"),
        "goal_rate": mean("goal_reached"),
        "reverse_fraction": mean("reverse_fraction"),
        "pct_steps_in_collision": mean("pct_steps_in_collision"),
        "exploration": mean("exploration"),
        "wall_side": mean("wall_side"),
        "light_reliance": light_reliance(network, inputs),
    }


def evaluate_genome(genome, maze_seed, noise_seed_base):
    """Evaluate one genome on the batch's 8 mazes (regenerated from ``maze_seed``)."""
    network = make_ga().decode_genome(genome)
    mazes = sample_random_mazes(np.random.default_rng(maze_seed), MAZES_PER_BATCH)
    inputs, runs = [], []
    for index, rm in enumerate(mazes):
        runs.append(run_episode(network, rm.maze, rm.start, rm.start_heading,
                                noise_seed_base + index, inputs))
    return summarize_runs(network, runs, inputs)


def _evaluate_job(args):
    return evaluate_genome(*args)


class NetworkMapElites:
    """Batch-parallel MAP-Elites archive over (turning bias, speed)."""

    def __init__(self, bins=DEFAULT_BINS, speed_range=DEFAULT_SPEED_RANGE, seed=0, batch_size=45):
        if min(bins) < 1 or batch_size < 1 or speed_range[1] <= speed_range[0]:
            raise ValueError("bins and batch_size must be positive and speed_range increasing")
        self.bins = tuple(bins)
        self.speed_range = tuple(speed_range)
        self.batch_size = batch_size
        self.seed = seed
        self.ga = make_ga(seed)
        self.rng = np.random.default_rng([seed, 2])     # parent selection
        self.maze_rng = np.random.default_rng([seed, 3])
        self.archive = {}
        self.improvements = {}
        self.evaluations = 0
        self.batches = 0
        self.log = []

    def cell_for(self, turning_bias, speed):
        scaled = ((turning_bias - BIAS_RANGE[0]) / (BIAS_RANGE[1] - BIAS_RANGE[0]),
                  (speed - self.speed_range[0]) / (self.speed_range[1] - self.speed_range[0]))
        return tuple(int(min(max(value, 0.0), 1.0 - 1e-9) * count)
                     for value, count in zip(scaled, self.bins))

    def selection_weights(self, cells):
        weights = []
        for column, row in cells:
            empty = sum(
                1 for dc in (-1, 0, 1) for dr in (-1, 0, 1)
                if (dc or dr) and 0 <= column + dc < self.bins[0] and 0 <= row + dr < self.bins[1]
                and (column + dc, row + dr) not in self.archive
            )
            weights.append((1 + empty) / (1 + self.improvements.get((column, row), 0)))
        weights = np.array(weights, dtype=float)
        return weights / weights.sum()

    def make_children(self, count):
        cells = sorted(self.archive)
        weights = self.selection_weights(cells)
        children = []
        for _ in range(count):
            first, second = self.rng.choice(len(cells), size=2, p=weights)
            child, _ = self.ga.crossover(self.archive[cells[first]]["genome"],
                                         self.archive[cells[second]]["genome"])
            children.append(self.ga.mutate(child))
        return children

    def insert(self, genome, result, source):
        self.evaluations += 1
        cell = self.cell_for(result["turning_bias"], result["speed"])
        incumbent = self.archive.get(cell)
        inserted = incumbent is None or result["fitness"] > incumbent["fitness"]
        if inserted:
            if incumbent is not None:
                self.improvements[cell] = self.improvements.get(cell, 0) + 1
            self.archive[cell] = {"genome": np.asarray(genome, dtype=float).copy(),
                                  "evaluation": self.evaluations, "source": source, **result}
        self.log.append({"evaluation": self.evaluations, "batch": self.batches, "source": source,
                         "cell_column": cell[0], "cell_row": cell[1], "inserted": int(inserted),
                         **result, **{f"gene_{i}": g for i, g in enumerate(genome)}})
        return inserted

    def run_batch(self, genomes, pool, source="child"):
        maze_seed = int(self.maze_rng.integers(2**31))
        noise_seed_base = int(self.maze_rng.integers(2**31))
        jobs = [(g, maze_seed, noise_seed_base) for g in genomes]
        results = pool.map(_evaluate_job, jobs) if pool is not None else [_evaluate_job(j) for j in jobs]
        inserted = sum(self.insert(g, r, source) for g, r in zip(genomes, results))
        self.batches += 1
        return inserted

    def stats(self):
        elites = list(self.archive.values())
        return {
            "filled": len(elites),
            "cells": self.bins[0] * self.bins[1],
            "best_fitness": max((e["fitness"] for e in elites), default=float("nan")),
            "mean_goal_rate": float(np.mean([e["goal_rate"] for e in elites])) if elites else float("nan"),
        }

    # ----- checkpoints -----
    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        state = {key: getattr(self, key) for key in (
            "bins", "speed_range", "batch_size", "seed", "archive", "improvements",
            "evaluations", "batches")}
        state["rng_states"] = (self.rng.bit_generator.state, self.maze_rng.bit_generator.state,
                               self.ga.rng.bit_generator.state)
        tmp = out_dir / "checkpoint.pkl.tmp"
        with tmp.open("wb") as handle:
            pickle.dump(state, handle)
        os.replace(tmp, out_dir / "checkpoint.pkl")
        cells = sorted(self.archive)
        np.savez(out_dir / "archive.npz", cells=np.array(cells),
                 genomes=np.array([self.archive[c]["genome"] for c in cells]),
                 bins=self.bins, speed_range=self.speed_range,
                 **{name: np.array([self.archive[c][name] for c in cells])
                    for name in ("fitness", "turning_bias", "speed") + LOGGED_STATS})
        if self.log:
            path = out_dir / "evaluations.csv"
            new = not path.exists()
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.log[0]))
                if new:
                    writer.writeheader()
                writer.writerows(self.log)
            self.log = []

    @classmethod
    def load(cls, out_dir):
        with (Path(out_dir) / "checkpoint.pkl").open("rb") as handle:
            state = pickle.load(handle)
        search = cls(state["bins"], state["speed_range"], state["seed"], state["batch_size"])
        for key in ("archive", "improvements", "evaluations", "batches"):
            setattr(search, key, state[key])
        search.rng.bit_generator.state, search.maze_rng.bit_generator.state, \
            search.ga.rng.bit_generator.state = state["rng_states"]
        return search


def search(out_dir, total_evaluations=10000, seed_genomes=None, workers=9, batch_size=45,
           speed_range=DEFAULT_SPEED_RANGE, checkpoint_every=450, resume=False, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        archive = NetworkMapElites.load(out_dir)
    else:
        archive = NetworkMapElites(speed_range=speed_range, seed=seed, batch_size=batch_size)
    progress = (out_dir / "progress.txt").open("a")
    start, start_evaluations = time.time(), archive.evaluations
    last_checkpoint = archive.evaluations
    with Pool(workers) as pool:
        if not archive.archive:
            archive.run_batch(list(seed_genomes), pool, source="seed")
            archive.save(out_dir)
        while archive.evaluations < total_evaluations:
            count = min(archive.batch_size, total_evaluations - archive.evaluations)
            inserted = archive.run_batch(archive.make_children(count), pool)
            elapsed = time.time() - start
            rate = (archive.evaluations - start_evaluations) / max(elapsed, 1e-9) * 60
            remaining = (total_evaluations - archive.evaluations) / max(rate, 1e-9)
            stats = archive.stats()
            print(f"{time.strftime('%H:%M:%S')} eval {archive.evaluations} batch {archive.batches} "
                  f"inserted {inserted}/{count} filled {stats['filled']}/{stats['cells']} "
                  f"best {stats['best_fitness']:.3f} mean goal {stats['mean_goal_rate']:.2f} "
                  f"rate {rate:.0f}/min eta {remaining:.0f} min", file=progress, flush=True)
            if archive.evaluations - last_checkpoint >= checkpoint_every \
                    or archive.evaluations >= total_evaluations:
                archive.save(out_dir)
                last_checkpoint = archive.evaluations
    progress.close()
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("--seed-genomes", help="npz of genomes to start from (one array per key)")
    parser.add_argument("--evaluations", type=int, default=10000,
                        help="total evaluations including the seed genomes")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=45)
    parser.add_argument("--speed-range", type=float, nargs=2, default=DEFAULT_SPEED_RANGE)
    parser.add_argument("--checkpoint-every", type=int, default=450)
    parser.add_argument("--resume", action="store_true", help="continue from out_dir/checkpoint.pkl")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    genomes = None
    if not arguments.resume:
        if not arguments.seed_genomes:
            parser.error("--seed-genomes is required for a new search")
        with np.load(arguments.seed_genomes) as data:
            genomes = [data[key] for key in sorted(data.files, key=lambda k: (len(k), k))]
    search(arguments.out_dir, arguments.evaluations, genomes, arguments.workers,
           arguments.batch_size, tuple(arguments.speed_range), arguments.checkpoint_every,
           arguments.resume, arguments.seed)


if __name__ == "__main__":
    main()
