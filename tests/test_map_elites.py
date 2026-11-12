import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evolution.map_elites import MapElites, load_seed_genomes, personality_representatives, plot_archive
from evolution.fitness import evaluate_fitness
from sim.maze import Maze
from sim.robot import Vehicle
from sim.world import World


class MapElitesTests(unittest.TestCase):
    def test_seed_archive_re_evaluates_genomes_and_counts_budget(self):
        source = MapElites(seed=1, behavior_mode="personality", noise_stdev=0)
        source.run(evaluations=4, initial=4, steps=20, log_every=100)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seeds.npz"
            source.save(path)
            genomes = load_seed_genomes(path, source.ga.genome_size)
            target = MapElites(seed=2, behavior_mode="personality", noise_stdev=0)
            target.run(evaluations=len(genomes) + 2, initial=0, steps=40,
                       log_every=100, seed_genomes=genomes)
            self.assertEqual(target.seed_count, len(genomes))
            self.assertEqual(target.evaluations, len(genomes) + 2)
            for genome, (_, _, _, elite) in zip(genomes, target.evaluation_log):
                np.testing.assert_array_equal(genome, elite["genome"])
                np.testing.assert_array_equal(elite["path"], target.evaluate(genome)["path"])
            target.save(path)
            with np.load(path) as data:
                self.assertEqual(int(data["seed_count"]), len(genomes))
                self.assertEqual(int(data["steps"]), 40)

    def test_invalid_seed_archives_and_budget_are_rejected(self):
        archive = MapElites()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.npz"
            for genomes in (
                np.empty((0, archive.ga.genome_size)),
                np.zeros((2, archive.ga.genome_size + 1)),
                np.full((1, archive.ga.genome_size), np.nan),
            ):
                np.savez(path, genomes=genomes)
                with self.assertRaises(ValueError):
                    load_seed_genomes(path, archive.ga.genome_size)
        with self.assertRaises(ValueError):
            archive.run(evaluations=1, seed_genomes=np.zeros((2, archive.ga.genome_size)))
        self.assertEqual(archive.evaluations, 0)

    def test_cli_writes_default_results_and_respects_custom_output(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(project), env.get("PYTHONPATH")))
            )
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable, "-m", "evolution.map_elites",
                "--personality", "--evaluations", "2", "--initial", "2",
                "--steps", "5", "--output", "custom.png",
            ]
            subprocess.run(command, cwd=folder, env=env, check=True, capture_output=True)
            results = Path(folder) / "results"
            self.assertTrue((Path(folder) / "custom.png").is_file())
            self.assertFalse((Path(folder) / "test_output" / "map_elites_archive.png").exists())
            for name in (
                "map_elites_archive.npz", "map_elites_evaluations.csv",
                "map_elites_elites.csv",
            ):
                self.assertTrue((results / name).is_file())
            command = command[:-2]
            subprocess.run(command, cwd=folder, env=env, check=True, capture_output=True)
            self.assertTrue((Path(folder) / "test_output" / "map_elites_archive.png").is_file())

    def test_cell_mapping_clips_to_grid_edges(self):
        archive = MapElites(bins=(10, 5), coverage_range=(0.0, 0.2), progress_max=1.0)

        self.assertEqual(archive.cell_for((0.0, 0.0)), (0, 0))
        self.assertEqual(archive.cell_for((0.1, 0.5)), (5, 2))
        self.assertEqual(archive.cell_for((5.0, 1.0)), (9, 4))
        self.assertEqual(archive.cell_for((-1.0, -1.0)), (0, 0))

    def test_cell_keeps_only_its_best_controller(self):
        archive = MapElites(seed=1)
        archive.steps = 50
        scores = iter((1.0, 5.0, 3.0))
        archive.evaluate = lambda genome: {
            "genome": genome, "fitness": next(scores), "descriptor": (0.1, 0.5),
        }

        results = [archive.add(np.zeros(3)) for _ in range(3)]

        self.assertEqual(results, [True, True, False])
        self.assertEqual(len(archive.archive), 1)
        self.assertEqual(next(iter(archive.archive.values()))["fitness"], 5.0)

    def test_run_populates_archive_and_figure_is_saved(self):
        archive = MapElites(seed=2, noise_stdev=0.0)
        archive.run(evaluations=12, initial=6, steps=150, log_every=1000)

        self.assertEqual(archive.evaluations, 12)
        self.assertGreaterEqual(len(archive.archive), 1)
        with tempfile.TemporaryDirectory() as folder:
            figure = Path(folder) / "archive.png"
            archive.save(Path(folder) / "archive.npz")
            plot_archive(archive, figure)
            self.assertGreater(figure.stat().st_size, 0)

    def test_csv_logs_every_evaluation_and_elite_with_genomes(self):
        archive = MapElites(seed=2, noise_stdev=0.0)
        archive.run(evaluations=10, initial=5, steps=100, log_every=1000)
        with tempfile.TemporaryDirectory() as folder:
            evaluations = Path(folder) / "evaluations.csv"
            elites = Path(folder) / "elites.csv"
            archive.save_csv(evaluations, evaluated=True)
            archive.save_csv(elites, evaluated=False)
            with evaluations.open() as handle:
                evaluation_rows = list(csv.DictReader(handle))
            with elites.open() as handle:
                elite_rows = list(csv.DictReader(handle))

        self.assertEqual(len(evaluation_rows), 10)
        self.assertEqual(len(elite_rows), len(archive.archive))
        self.assertIn(f"gene_{archive.ga.genome_size - 1}", evaluation_rows[0])
        self.assertGreaterEqual(
            sum(int(row["inserted"]) for row in evaluation_rows),
            len(archive.archive),
        )

    def test_search_is_reproducible(self):
        runs = []
        for _ in range(2):
            archive = MapElites(seed=3)
            archive.run(evaluations=8, initial=4, steps=100, log_every=1000)
            runs.append(sorted((c, e["fitness"]) for c, e in archive.archive.items()))

        self.assertEqual(runs[0], runs[1])

    def test_personality_cells_and_labels_use_both_dimensions(self):
        archive = MapElites(
            bins=(10, 10), coverage_range=(0.0, 0.2),
            progress_max=0.25, behavior_mode="personality",
        )
        examples = (
            ((0.8, 0), 0.05, True, (8, 0), "collision-free direct"),
            ((0.8, 2), 0.05, True, (8, 2), "collision-tolerant direct"),
            ((0.3, 0), 0.15, True, (3, 0), "collision-free explorer"),
            ((0.3, 12), 0.15, True, (3, 9), "collision-tolerant explorer"),
            ((0.8, 0), 0.05, False, (8, 0), "collision-free limited exploration (unfinished)"),
            ((0.3, 2), 0.15, False, (3, 2), "collision-tolerant explorer (unfinished)"),
        )
        for descriptor, coverage, reached_goal, cell, label in examples:
            with self.subTest(descriptor=descriptor):
                self.assertEqual(archive.cell_for(descriptor), cell)
                self.assertEqual(
                    archive.personality_for(*descriptor, coverage, reached_goal), label
                )

    def test_personality_descriptor_matches_measured_run(self):
        archive = MapElites(seed=7, noise_stdev=0.0, behavior_mode="personality")
        archive.steps = 100
        genome = archive.ga._random_genome()
        elite = archive.evaluate(genome)
        world = World(
            maze=Maze(difficulty=archive.difficulty),
            vehicle=Vehicle(
                controller=archive.ga.decode_genome(genome),
                noise_stdev=archive.noise_stdev,
                noise_seed=archive.noise_seed,
                navigation_heuristics=archive.navigation_heuristics,
            ),
        )
        world.run(archive.steps)
        metrics = evaluate_fitness(world)

        expected_directness = (
            min(1.0, max(0.0, metrics["progress"]) / metrics["path_distance"])
            if metrics["path_distance"] else 0.0
        )
        self.assertEqual(elite["descriptor"], (expected_directness, metrics["collisions"]))
        self.assertEqual(elite["goal_progress"], metrics["normalized_goal_progress"])
        self.assertEqual(elite["coverage"], metrics["coverage"])
        expected_score = max(
            metrics["score"] - metrics["score_components"]["collision"]
            - metrics["score_components"]["repeated_collision"], -100.0,
        ) + (100.0 if metrics["reached_goal"] else 0.0) + elite["exploration_adjustment"]
        self.assertAlmostEqual(elite["fitness"], expected_score)
        self.assertEqual(elite["path_revisits"], metrics["path_revisits"])
        self.assertEqual(elite["unique_cells_visited"], metrics["unique_cells_visited"])

    def test_personality_outputs_preserve_descriptor_and_progress(self):
        archive = MapElites(seed=2, noise_stdev=0.0, behavior_mode="personality")
        archive.run(evaluations=8, initial=4, steps=100, log_every=1000)
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / "archive.npz"
            csv_path = Path(folder) / "elites.csv"
            figure = Path(folder) / "archive.png"
            archive.save(saved)
            archive.save_csv(csv_path, evaluated=False)
            plot_archive(archive, figure)
            with np.load(saved) as data, csv_path.open() as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(str(data["behavior_mode"]), "personality")
                self.assertEqual(len(rows), len(data["descriptors"]))
                for index, row in enumerate(rows):
                    self.assertAlmostEqual(float(row["coverage"]), data["coverage"][index])
                    self.assertAlmostEqual(float(row["directness"]), data["descriptors"][index, 0])
                    self.assertEqual(int(row["collision_count"]), data["descriptors"][index, 1])
                    self.assertAlmostEqual(float(row["safe_distance_score"]), data["safe_distance_score"][index])
                    self.assertAlmostEqual(float(row["goal_progress"]), data["goal_progress"][index])
                    self.assertEqual(row["personality"], data["personalities"][index])
                    self.assertEqual(int(row["path_revisits"]), data["path_revisits"][index])
                    self.assertEqual(int(row["unique_cells_visited"]), data["unique_cells_visited"][index])
                    self.assertAlmostEqual(float(row["exploration_adjustment"]), data["exploration_adjustment"][index])
            self.assertGreater(figure.stat().st_size, 0)

    def test_explorer_selection_rewards_new_cells_without_extra_revisit_penalty(self):
        archive = MapElites(behavior_mode="personality", noise_stdev=0)
        archive.steps = 0
        genome = archive.ga._random_genome()
        base = evaluate_fitness(World(vehicle=Vehicle(noise_stdev=0)))
        base.update({
            "score": 0.0, "progress": 4.0, "path_distance": 20.0,
            "coverage": 0.1, "unique_cells_visited": 40,
            "path_revisits": 5, "collisions": 0, "reached_goal": True,
            "score_components": {"collision": 0.0, "repeated_collision": 0.0},
        })
        fewer_revisits = dict(base, path_revisits=1, score=2.0)
        more_new_cells = dict(fewer_revisits, unique_cells_visited=42)
        with patch("evolution.map_elites.evaluate_fitness",
                   side_effect=[base, fewer_revisits, more_new_cells]):
            self.assertTrue(archive.add(genome))
            first = next(iter(archive.archive.values()))
            self.assertTrue(archive.add(genome))
            second = next(iter(archive.archive.values()))
            self.assertTrue(archive.add(genome))
            third = next(iter(archive.archive.values()))
        self.assertEqual(first["personality"], "collision-free explorer")
        self.assertEqual(first["exploration_adjustment"], 80.0)
        self.assertEqual(second["exploration_adjustment"], 80.0)
        self.assertEqual(second["fitness"] - first["fitness"], 2.0)
        self.assertEqual(third["fitness"] - second["fitness"], 4.0)

    def test_explorer_adjustment_does_not_change_direct_or_progress_mode(self):
        metrics = evaluate_fitness(World(vehicle=Vehicle(noise_stdev=0)))
        metrics.update({
            "progress": 8.0, "path_distance": 10.0, "coverage": 0.1,
            "unique_cells_visited": 40, "path_revisits": 5, "reached_goal": True,
        })
        for mode in ("progress", "personality"):
            archive = MapElites(behavior_mode=mode, noise_stdev=0)
            archive.steps = 0
            with patch("evolution.map_elites.evaluate_fitness", return_value=metrics):
                elite = archive.evaluate(archive.ga._random_genome())
            self.assertEqual(elite["exploration_adjustment"], 0.0)

    def test_goal_reaching_elite_beats_higher_scoring_unfinished_run(self):
        archive = MapElites(behavior_mode="personality")
        candidates = iter((
            {"descriptor": (0.5, 1), "fitness": 500.0, "reached_goal": False},
            {"descriptor": (0.5, 1), "fitness": 5.0, "reached_goal": True},
            {"descriptor": (0.5, 1), "fitness": 1000.0, "reached_goal": False},
        ))
        archive.evaluate = lambda genome: next(candidates)
        self.assertEqual([archive.add(np.zeros(3)) for _ in range(3)], [True, True, False])
        self.assertTrue(next(iter(archive.archive.values()))["reached_goal"])

    def test_gallery_selects_one_best_path_per_personality(self):
        archive = MapElites(behavior_mode="personality")
        archive.archive = {
            (0, 0): {"personality": "collision-free explorer", "fitness": 10, "reached_goal": True},
            (1, 0): {"personality": "collision-free explorer", "fitness": 20, "reached_goal": True},
            (2, 0): {"personality": "collision-free direct", "fitness": 1, "reached_goal": True},
            (3, 0): {"personality": "collision-free direct", "fitness": 100, "reached_goal": False},
        }
        representatives = personality_representatives(archive)
        self.assertEqual([cell for cell, _ in representatives], [(1, 0), (2, 0)])

    def test_personality_default_run_has_longer_step_budget(self):
        archive = MapElites(behavior_mode="personality")
        archive.run(evaluations=0)
        self.assertEqual(archive.steps, 6000)
        archive.run(evaluations=0, steps=123)
        self.assertEqual(archive.steps, 123)

    def test_invalid_behavior_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            MapElites(behavior_mode="unknown")


if __name__ == "__main__":
    unittest.main()
