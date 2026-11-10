import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evolution.map_elites import MapElites, plot_archive


class MapElitesTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
