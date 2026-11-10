"""MAP-Elites quality-diversity search over neural maze controllers.

The archive is a grid over the behavior descriptor produced by
``evaluate_fitness``: (maze coverage, normalized goal progress). Each cell
keeps the single best controller found with that behavior.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle

from evolution.fitness import evaluate_fitness
from evolution.genetic_algorithm import GeneticAlgorithm
from sim.maze import Maze
from sim.robot import Vehicle
from sim.world import World

DEFAULT_BINS = (10, 10)
DEFAULT_COVERAGE_RANGE = (0.03, 0.2)
DEFAULT_PROGRESS_MAX = 0.85
SCORE_FLOOR = -100.0
GOAL_BONUS = 100.0
PATH_POINTS = 150


class MapElites:
    """Grid archive of elites indexed by (coverage, goal progress)."""

    def __init__(
        self,
        difficulty=1,
        bins=DEFAULT_BINS,
        coverage_range=DEFAULT_COVERAGE_RANGE,
        progress_max=DEFAULT_PROGRESS_MAX,
        hidden_size=4,
        mutation_rate=0.2,
        mutation_scale=0.3,
        noise_stdev=0.05,
        noise_seed=12345,
        navigation_heuristics=True,
        seed=0,
    ):
        if min(bins) < 1:
            raise ValueError("bins must be positive")
        if coverage_range[1] <= coverage_range[0] or progress_max <= 0:
            raise ValueError("descriptor ranges must be positive and increasing")
        self.difficulty = difficulty
        self.bins = tuple(bins)
        self.coverage_range = tuple(coverage_range)
        self.progress_max = progress_max
        self.noise_stdev = noise_stdev
        self.noise_seed = noise_seed
        self.navigation_heuristics = navigation_heuristics
        # The GA object supplies genome creation, mutation, crossover, and decoding.
        self.ga = GeneticAlgorithm(
            population_size=2,
            elite_count=0,
            tournament_size=1,
            hidden_size=hidden_size,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            seed=seed,
        )
        self.rng = self.ga.rng
        self.archive = {}
        self.evaluations = 0
        self.steps = 3000
        self.history = []

    def cell_for(self, descriptor):
        """Map a (coverage, progress) descriptor to integer grid coordinates."""
        coverage, progress = descriptor
        low, high = self.coverage_range
        scaled = ((coverage - low) / (high - low), progress / self.progress_max)
        return tuple(
            int(min(max(value, 0.0), 1.0 - 1e-9) * count)
            for value, count in zip(scaled, self.bins)
        )

    def evaluate(self, genome):
        vehicle = Vehicle(
            controller=self.ga.decode_genome(genome),
            noise_stdev=self.noise_stdev,
            noise_seed=self.noise_seed,
            navigation_heuristics=self.navigation_heuristics,
        )
        world = World(maze=Maze(difficulty=self.difficulty), vehicle=vehicle)
        world.run(self.steps)
        metrics = evaluate_fitness(world)
        score = max(metrics["score"], SCORE_FLOOR)
        if metrics["reached_goal"]:
            score += GOAL_BONUS
        history = np.asarray(world.position_history, dtype=float)
        indices = np.linspace(0, len(history) - 1, min(PATH_POINTS, len(history))).astype(int)
        return {
            "genome": np.asarray(genome, dtype=float).copy(),
            "fitness": float(score),
            "descriptor": tuple(metrics["behavior_descriptor"]),
            "reached_goal": bool(metrics["reached_goal"]),
            "collisions": int(metrics["collisions"]),
            "path_length": float(metrics["path_distance"]),
            "completion_steps": world.steps if metrics["reached_goal"] else None,
            "path": history[indices],
        }

    def add(self, genome):
        """Evaluate a genome and keep it if it is the best in its cell."""
        elite = self.evaluate(genome)
        self.evaluations += 1
        cell = self.cell_for(elite["descriptor"])
        incumbent = self.archive.get(cell)
        if incumbent is None or elite["fitness"] > incumbent["fitness"]:
            self.archive[cell] = elite
            return True
        return False

    def run(self, evaluations=1000, initial=50, steps=3000, log_every=100):
        self.steps = steps
        for _ in range(min(initial, evaluations)):
            self.add(self.ga._random_genome())
            self._log(log_every)
        while self.evaluations < evaluations:
            keys = list(self.archive)
            parent = self.archive[keys[self.rng.integers(len(keys))]]["genome"]
            if self.rng.random() < self.ga.crossover_rate:
                other = self.archive[keys[self.rng.integers(len(keys))]]["genome"]
                parent, _ = self.ga.crossover(parent, other)
            self.add(self.ga.mutate(parent))
            self._log(log_every)
        return self.archive

    def _log(self, every):
        if self.evaluations % every == 0:
            stats = self.stats()
            self.history.append((self.evaluations, stats["coverage"], stats["best_fitness"]))
            print(
                f"  eval {self.evaluations:5d}: "
                f"{stats['filled']}/{stats['cells']} cells "
                f"({stats['coverage']:.0%}), best fitness {stats['best_fitness']:.1f}, "
                f"{stats['solving_cells']} goal-reaching cells"
            )

    def stats(self):
        cells = self.bins[0] * self.bins[1]
        elites = list(self.archive.values())
        return {
            "filled": len(elites),
            "cells": cells,
            "coverage": len(elites) / cells,
            "best_fitness": max((e["fitness"] for e in elites), default=float("nan")),
            "solving_cells": sum(e["reached_goal"] for e in elites),
        }

    def save(self, path):
        """Store every elite's genome and behavior so controllers can be replayed."""
        cells = sorted(self.archive)
        np.savez(
            path,
            cells=np.array(cells),
            genomes=np.array([self.archive[c]["genome"] for c in cells]),
            fitness=np.array([self.archive[c]["fitness"] for c in cells]),
            descriptors=np.array([self.archive[c]["descriptor"] for c in cells]),
            reached_goal=np.array([self.archive[c]["reached_goal"] for c in cells]),
            difficulty=self.difficulty,
            hidden_size=self.ga.hidden_size,
        )


def _draw_maze(axis, maze):
    for (x0, y0), (x1, y1) in maze.interior_walls:
        axis.plot([x0, x1], [y0, y1], color="black", linewidth=1.5)
    for x0, y0, x1, y1 in maze.obstacles:
        axis.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color="dimgray")
    axis.add_patch(Circle(maze.goal, maze.goal_radius, color="limegreen", alpha=0.6))
    axis.set_xlim(maze.x_min, maze.x_max)
    axis.set_ylim(maze.y_min, maze.y_max)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def plot_archive(archive, output_path="map_elites_archive.png", show=False):
    """Draw the heatmap of cell fitness and a gallery of each elite's path."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns, rows = archive.bins
    grid = np.full((rows, columns), np.nan)
    for (column, row), elite in archive.archive.items():
        grid[row, column] = elite["fitness"]

    figure = plt.figure(figsize=(15, 7.5))
    heat_axis = figure.add_subplot(1, 2, 1)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    image = heat_axis.imshow(
        grid, origin="lower", cmap=cmap, aspect="auto",
        extent=(*archive.coverage_range, 0, archive.progress_max),
    )
    figure.colorbar(image, ax=heat_axis, label="Best fitness in cell")
    width = (archive.coverage_range[1] - archive.coverage_range[0]) / columns
    height = archive.progress_max / rows
    for (column, row), elite in archive.archive.items():
        center = (archive.coverage_range[0] + (column + 0.5) * width, (row + 0.5) * height)
        if elite["reached_goal"]:
            heat_axis.plot(*center, marker="*", color="gold", markersize=14,
                           markeredgecolor="black")
        heat_axis.text(center[0], center[1] - height * 0.28, f"{elite['fitness']:.0f}",
                       ha="center", va="center", fontsize=6, color="white")
    stats = archive.stats()
    heat_axis.set_xlabel("Maze coverage (exploration)")
    heat_axis.set_ylabel("Normalized goal progress")
    heat_axis.set_title(
        f"MAP-Elites archive, course {archive.difficulty}: "
        f"{stats['filled']}/{stats['cells']} cells filled "
        f"({stats['solving_cells']} reach goal, marked *)"
    )

    ranked = sorted(archive.archive.items(), key=lambda item: -item[1]["fitness"])[:12]
    gallery = figure.add_gridspec(3, 4, left=0.55, right=0.98, bottom=0.04, top=0.9,
                                  wspace=0.08, hspace=0.35)
    maze = Maze(difficulty=archive.difficulty)
    for index, ((column, row), elite) in enumerate(ranked):
        axis = figure.add_subplot(gallery[index // 4, index % 4])
        _draw_maze(axis, maze)
        axis.plot(elite["path"][:, 0], elite["path"][:, 1],
                  color="crimson" if not elite["reached_goal"] else "royalblue", linewidth=1.2)
        axis.set_title(
            f"cell ({column},{row}) fit {elite['fitness']:.0f}"
            + (" GOAL" if elite["reached_goal"] else ""),
            fontsize=7,
        )
    figure.text(0.76, 0.94, "Best controller paths (top 12 cells by fitness)",
                ha="center", fontsize=10)
    figure.savefig(output_path, dpi=150)
    print(f"Archive figure saved to {Path(output_path).resolve()}")
    if show:
        plt.show()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--evaluations", type=int, default=1500)
    parser.add_argument("--initial", type=int, default=100)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--bins", type=int, nargs=2, default=DEFAULT_BINS)
    parser.add_argument("--coverage-range", type=float, nargs=2, default=DEFAULT_COVERAGE_RANGE)
    parser.add_argument("--progress-max", type=float, default=DEFAULT_PROGRESS_MAX)
    parser.add_argument("--noise-stdev", type=float, default=0.05)
    parser.add_argument("--neural-only", action="store_true",
                        help="disable the hard-coded navigation heuristics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="map_elites_archive.png")
    parser.add_argument("--save", default="map_elites_archive.npz")
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()

    archive = MapElites(
        difficulty=arguments.course,
        bins=arguments.bins,
        coverage_range=arguments.coverage_range,
        progress_max=arguments.progress_max,
        noise_stdev=arguments.noise_stdev,
        navigation_heuristics=not arguments.neural_only,
        seed=arguments.seed,
    )
    print(f"Running MAP-Elites on course {arguments.course}...")
    archive.run(
        evaluations=arguments.evaluations,
        initial=arguments.initial,
        steps=arguments.steps,
    )
    archive.save(arguments.save)
    plot_archive(archive, arguments.output, show=arguments.show)


if __name__ == "__main__":
    main()
