import tempfile
import unittest
from unittest import mock

import numpy as np

from controller.network import NeuralNetwork
from evolution import network_map_elites as nme


def fake_result(genome):
    return {"fitness": float(genome[0]), "turning_bias": float(np.tanh(genome[1])),
            "speed": 0.35 + 0.28 * (0.5 + 0.5 * np.tanh(genome[2])),
            **{name: 0.0 for name in nme.LOGGED_STATS}}


def fake_job(args):
    return fake_result(args[0])


class ArchiveTests(unittest.TestCase):
    def test_cells_cover_both_axes_and_clip(self):
        archive = nme.NetworkMapElites()
        self.assertEqual(archive.cell_for(-1.0, 0.35), (0, 0))
        self.assertEqual(archive.cell_for(1.0, 0.63), (9, 9))
        self.assertEqual(archive.cell_for(0.05, 0.10), (5, 0))     # slower than range -> edge bin
        self.assertEqual(archive.cell_for(-0.05, 0.99), (4, 9))

    def test_insert_keeps_the_better_controller(self):
        archive = nme.NetworkMapElites()
        genome = np.zeros(archive.ga.genome_size)
        result = fake_result(genome)
        self.assertTrue(archive.insert(genome, {**result, "fitness": 0.5}, "seed"))
        self.assertFalse(archive.insert(genome, {**result, "fitness": 0.4}, "child"))
        self.assertTrue(archive.insert(genome, {**result, "fitness": 0.6}, "child"))
        cell = archive.cell_for(result["turning_bias"], result["speed"])
        self.assertEqual(archive.improvements[cell], 1)
        self.assertEqual(archive.evaluations, 3)

    def test_selection_favors_sparse_and_rarely_improved_cells(self):
        archive = nme.NetworkMapElites()
        elite = {"genome": np.zeros(archive.ga.genome_size), "fitness": 0.0}
        for cell in [(0, 0), (1, 0), (0, 1), (1, 1), (5, 5)]:
            archive.archive[cell] = elite
        weights = dict(zip(sorted(archive.archive), archive.selection_weights(sorted(archive.archive))))
        self.assertGreater(weights[(5, 5)], weights[(0, 0)])     # isolated beats crowded corner
        archive.improvements[(5, 5)] = 10
        weights = dict(zip(sorted(archive.archive), archive.selection_weights(sorted(archive.archive))))
        self.assertLess(weights[(5, 5)], weights[(0, 0)])        # often improved -> picked less

    @mock.patch.object(nme, "_evaluate_job", fake_job)
    def test_batches_checkpoint_and_resume(self):
        rng = np.random.default_rng(0)
        archive = nme.NetworkMapElites(batch_size=10)
        seeds = [rng.normal(size=archive.ga.genome_size) for _ in range(5)]
        archive.run_batch(seeds, pool=None, source="seed")
        archive.run_batch(archive.make_children(10), pool=None)
        with tempfile.TemporaryDirectory() as folder:
            archive.save(folder)
            restored = nme.NetworkMapElites.load(folder)
            self.assertEqual(restored.evaluations, 15)
            self.assertEqual(sorted(restored.archive), sorted(archive.archive))
            a = archive.make_children(3)
            b = restored.make_children(3)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)                 # random streams restored too
            data = np.load(f"{folder}/archive.npz")
            self.assertEqual(len(data["genomes"]), len(archive.archive))
            self.assertIn("light_reliance", data.files)


class EvaluationTests(unittest.TestCase):
    def test_light_reliance(self):
        network = NeuralNetwork(seed=1, input_size=5)
        inputs = np.random.default_rng(0).random((50, 5))
        self.assertGreater(nme.light_reliance(network, inputs), 0.0)
        inputs[:, 3:] = 0.0
        self.assertEqual(nme.light_reliance(network, inputs), 0.0)   # no light seen -> no change
        network.weights_input_hidden[3:, :] = 0.0
        self.assertEqual(nme.light_reliance(network, np.random.default_rng(1).random((50, 5))), 0.0)

    def test_real_evaluation_is_deterministic_and_in_range(self):
        genome = nme.make_ga(3).population[0].genome
        a = nme.evaluate_genome(genome, maze_seed=11, noise_seed_base=5)
        b = nme.evaluate_genome(genome, maze_seed=11, noise_seed_base=5)
        self.assertEqual(a, b)
        self.assertTrue(-1.0 <= a["turning_bias"] <= 1.0)
        self.assertTrue(0.0 <= a["speed"] <= 1.0)
        self.assertTrue(0.0 <= a["reverse_fraction"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
