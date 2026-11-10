"""MAP-Elites quality-diversity search over neural maze controllers.

The archive keeps the best controller in each behavior cell. The default grid
uses maze coverage and goal progress; personality mode uses route directness
and collision count instead.
"""

import argparse
import csv
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
DEFAULT_PERSONALITY_STEPS = 6000
DIRECTNESS_THRESHOLD = 0.4
EXPLORATION_THRESHOLD = 0.08
EXPLORER_NEW_CELL_REWARD = 2.0
DEFAULT_OUTPUT_DIR = Path("results")


class MapElites:
    """Grid archive indexed by progress or personality behavior descriptors."""

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
        behavior_mode="progress",
    ):
        if min(bins) < 1:
            raise ValueError("bins must be positive")
        if coverage_range[1] <= coverage_range[0] or progress_max <= 0:
            raise ValueError("descriptor ranges must be positive and increasing")
        if behavior_mode not in ("progress", "personality"):
            raise ValueError("behavior_mode must be 'progress' or 'personality'")
        self.difficulty = difficulty
        self.bins = tuple(bins)
        self.coverage_range = tuple(coverage_range)
        self.progress_max = progress_max
        self.behavior_mode = behavior_mode
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
        self.evaluation_log = []

    def cell_for(self, descriptor):
        """Map a behavior descriptor to integer grid coordinates."""
        first, second = descriptor
        if self.behavior_mode == "personality":
            return (
                int(min(max(first, 0.0), 1.0 - 1e-9) * self.bins[0]),
                min(max(int(second), 0), self.bins[1] - 1),
            )
        low, high = self.coverage_range
        scaled = ((first - low) / (high - low), second / self.progress_max)
        return tuple(
            int(min(max(value, 0.0), 1.0 - 1e-9) * count)
            for value, count in zip(scaled, self.bins)
        )

    def personality_for(self, directness, collisions, coverage, reached_goal):
        """Label observed route and collisions, not the controller's intent."""
        if reached_goal and directness >= DIRECTNESS_THRESHOLD:
            route = "direct"
        elif coverage >= EXPLORATION_THRESHOLD:
            route = "explorer"
        else:
            route = "limited exploration"
        contact = "collision-free" if collisions == 0 else "collision-tolerant"
        return f"{contact} {route}" + ("" if reached_goal else " (unfinished)")

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
        directness = (
            min(1.0, max(0.0, metrics["progress"]) / metrics["path_distance"])
            if metrics["path_distance"] > 0 else 0.0
        )
        personality = (
            self.personality_for(
                directness, metrics["collisions"], metrics["coverage"], metrics["reached_goal"]
            ) if self.behavior_mode == "personality" else ""
        )
        exploration_adjustment = 0.0
        if self.behavior_mode == "personality" and " explorer" in personality:
            exploration_adjustment = EXPLORER_NEW_CELL_REWARD * metrics["unique_cells_visited"]
        score = metrics["score"]
        if self.behavior_mode == "personality":
            score -= metrics["score_components"]["collision"]
            score -= metrics["score_components"]["repeated_collision"]
        score = max(score, SCORE_FLOOR)
        if metrics["reached_goal"]:
            score += GOAL_BONUS
        score += exploration_adjustment
        descriptor = (
            (directness, metrics["collisions"])
            if self.behavior_mode == "personality"
            else tuple(metrics["behavior_descriptor"])
        )
        history = np.asarray(world.position_history, dtype=float)
        indices = np.linspace(0, len(history) - 1, min(PATH_POINTS, len(history))).astype(int)
        return {
            "genome": np.asarray(genome, dtype=float).copy(),
            "fitness": float(score),
            "descriptor": descriptor,
            "coverage": float(metrics["coverage"]),
            "unique_cells_visited": int(metrics["unique_cells_visited"]),
            "path_revisits": int(metrics["path_revisits"]),
            "exploration_adjustment": float(exploration_adjustment),
            "directness": float(directness),
            "goal_progress": float(metrics["normalized_goal_progress"]),
            "safe_distance_score": float(metrics["safe_distance_score"]),
            "personality": personality,
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
        if self.behavior_mode == "personality":
            rank = lambda entry: (entry["reached_goal"], entry["fitness"])
        else:
            rank = lambda entry: entry["fitness"]
        inserted = incumbent is None or rank(elite) > rank(incumbent)
        self.evaluation_log.append((self.evaluations, cell, inserted, elite))
        if inserted:
            self.archive[cell] = elite
        return inserted

    def run(self, evaluations=1000, initial=50, steps=None, log_every=100):
        if steps is None:
            steps = DEFAULT_PERSONALITY_STEPS if self.behavior_mode == "personality" else 3000
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

    def save_csv(self, path, evaluated=True):
        """Write one row per evaluation (or per archive elite) with its genome.

        Each row has the behavior cell, descriptor, fitness, outcome metrics,
        and the genome as columns ``gene_0``..``gene_N``, so the file can be
        loaded directly as training data or to rebuild controllers.
        """
        if evaluated:
            rows = self.evaluation_log
        else:
            rows = [
                (None, cell, True, elite)
                for cell, elite in sorted(self.archive.items())
            ]
        gene_names = [f"gene_{i}" for i in range(self.ga.genome_size)]
        fields = [
            "evaluation", "course", "cell_column", "cell_row", "inserted",
            "coverage", "goal_progress", "directness", "safe_distance_score", "personality",
            "unique_cells_visited", "path_revisits", "exploration_adjustment",
            "fitness", "goal_success",
            "collision_count", "completion_steps", "path_length", *gene_names,
        ]
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for evaluation, cell, inserted, elite in rows:
                writer.writerow({
                    "evaluation": evaluation,
                    "course": self.difficulty,
                    "cell_column": cell[0],
                    "cell_row": cell[1],
                    "inserted": int(inserted),
                    "coverage": elite["coverage"],
                    "goal_progress": elite["goal_progress"],
                    "directness": elite["directness"],
                    "safe_distance_score": elite["safe_distance_score"],
                    "personality": elite["personality"],
                    "unique_cells_visited": elite["unique_cells_visited"],
                    "path_revisits": elite["path_revisits"],
                    "exploration_adjustment": elite["exploration_adjustment"],
                    "fitness": elite["fitness"],
                    "goal_success": int(elite["reached_goal"]),
                    "collision_count": elite["collisions"],
                    "completion_steps": (
                        "" if elite["completion_steps"] is None
                        else elite["completion_steps"]
                    ),
                    "path_length": elite["path_length"],
                    **dict(zip(gene_names, elite["genome"])),
                })

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
            behavior_mode=self.behavior_mode,
            coverage_range=self.coverage_range,
            progress_max=self.progress_max,
            noise_stdev=self.noise_stdev,
            noise_seed=self.noise_seed,
            navigation_heuristics=self.navigation_heuristics,
            steps=self.steps,
            bins=self.bins,
            coverage=np.array([self.archive[c]["coverage"] for c in cells]),
            unique_cells_visited=np.array([self.archive[c]["unique_cells_visited"] for c in cells]),
            path_revisits=np.array([self.archive[c]["path_revisits"] for c in cells]),
            exploration_adjustment=np.array([self.archive[c]["exploration_adjustment"] for c in cells]),
            directness=np.array([self.archive[c]["directness"] for c in cells]),
            collisions=np.array([self.archive[c]["collisions"] for c in cells]),
            path_length=np.array([self.archive[c]["path_length"] for c in cells]),
            goal_progress=np.array([self.archive[c]["goal_progress"] for c in cells]),
            safe_distance_score=np.array([self.archive[c]["safe_distance_score"] for c in cells]),
            personalities=np.array([self.archive[c]["personality"] for c in cells]),
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


def personality_representatives(archive):
    """Return the highest-ranked elite for each distinct observed personality."""
    ranked = sorted(
        archive.archive.items(),
        key=lambda item: (item[1]["reached_goal"], item[1]["fitness"]),
        reverse=True,
    )
    representatives = {}
    for cell, elite in ranked:
        representatives.setdefault(elite["personality"], (cell, elite))
    return list(representatives.values())


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
    personality = archive.behavior_mode == "personality"
    extent = (
        (0, 1, -0.5, rows - 0.5) if personality
        else (*archive.coverage_range, 0, archive.progress_max)
    )
    image = heat_axis.imshow(
        grid, origin="lower", cmap=cmap, aspect="auto",
        extent=extent,
    )
    figure.colorbar(image, ax=heat_axis, label="Selected fitness in cell")
    width = (extent[1] - extent[0]) / columns
    height = (extent[3] - extent[2]) / rows
    for (column, row), elite in archive.archive.items():
        center = (extent[0] + (column + 0.5) * width, extent[2] + (row + 0.5) * height)
        if elite["reached_goal"]:
            heat_axis.plot(*center, marker="*", color="gold", markersize=14,
                           markeredgecolor="black")
        heat_axis.text(center[0], center[1] - height * 0.28, f"{elite['fitness']:.0f}",
                       ha="center", va="center", fontsize=6, color="white")
    stats = archive.stats()
    heat_axis.set_xlabel("Route directness" if personality else "Maze coverage (exploration)")
    heat_axis.set_ylabel("Collision count" if personality else "Normalized goal progress")
    if personality:
        heat_axis.set_yticks(range(rows))
        heat_axis.set_yticklabels([str(i) for i in range(rows - 1)] + [f"{rows - 1}+"])
    heat_axis.set_title(
        f"MAP-Elites archive, course {archive.difficulty}: "
        f"{stats['filled']}/{stats['cells']} cells filled "
        f"({stats['solving_cells']} reach goal, marked *)"
    )

    ranked = (
        personality_representatives(archive) if personality
        else sorted(archive.archive.items(), key=lambda item: -item[1]["fitness"])[:12]
    )
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
            + (" GOAL" if elite["reached_goal"] else "")
            + (
                f"\n{elite['personality']}\n{elite['collisions']} collisions, "
                f"{elite['unique_cells_visited']} cells, {elite['path_revisits']} revisits"
                if personality else ""
            ),
            fontsize=7,
        )
    gallery_title = (
        "Best path per observed personality"
        if personality
        else "Best controller paths (top 12 cells by fitness)"
    )
    figure.text(0.76, 0.94, gallery_title,
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
    parser.add_argument("--steps", type=int, default=None,
                        help="cycles per run (default: 3000, or 6000 with --personality)")
    parser.add_argument("--bins", type=int, nargs=2, default=DEFAULT_BINS)
    parser.add_argument("--coverage-range", type=float, nargs=2, default=DEFAULT_COVERAGE_RANGE)
    parser.add_argument("--progress-max", type=float, default=DEFAULT_PROGRESS_MAX)
    parser.add_argument("--personality", action="store_true",
                        help="map route directness and collisions instead of coverage and progress")
    parser.add_argument("--noise-stdev", type=float, default=0.05)
    parser.add_argument("--neural-only", action="store_true",
                        help="disable the hard-coded navigation heuristics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "map_elites_archive.png"))
    parser.add_argument("--save", default=str(DEFAULT_OUTPUT_DIR / "map_elites_archive.npz"))
    parser.add_argument("--evaluations-csv", default=str(DEFAULT_OUTPUT_DIR / "map_elites_evaluations.csv"),
                        help="every evaluated controller with its genome")
    parser.add_argument("--elites-csv", default=str(DEFAULT_OUTPUT_DIR / "map_elites_elites.csv"),
                        help="the final archive: best controller per cell")
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()

    default_paths = {
        DEFAULT_OUTPUT_DIR / "map_elites_archive.png",
        DEFAULT_OUTPUT_DIR / "map_elites_archive.npz",
        DEFAULT_OUTPUT_DIR / "map_elites_evaluations.csv",
        DEFAULT_OUTPUT_DIR / "map_elites_elites.csv",
    }
    if any(Path(path) in default_paths for path in (
        arguments.output, arguments.save, arguments.evaluations_csv, arguments.elites_csv,
    )):
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    archive = MapElites(
        difficulty=arguments.course,
        bins=arguments.bins,
        coverage_range=arguments.coverage_range,
        progress_max=arguments.progress_max,
        noise_stdev=arguments.noise_stdev,
        navigation_heuristics=not arguments.neural_only,
        seed=arguments.seed,
        behavior_mode="personality" if arguments.personality else "progress",
    )
    print(f"Running MAP-Elites on course {arguments.course}...")
    archive.run(
        evaluations=arguments.evaluations,
        initial=arguments.initial,
        steps=arguments.steps,
    )
    archive.save(arguments.save)
    archive.save_csv(arguments.evaluations_csv, evaluated=True)
    archive.save_csv(arguments.elites_csv, evaluated=False)
    print(
        f"Saved {arguments.save}, {arguments.evaluations_csv} "
        f"({len(archive.evaluation_log)} evaluations), {arguments.elites_csv} "
        f"({len(archive.archive)} elites)"
    )
    plot_archive(archive, arguments.output, show=arguments.show)


if __name__ == "__main__":
    main()
