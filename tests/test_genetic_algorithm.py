import unittest

from evolution.genetic_algorithm import GeneticAlgorithm


class GeneticAlgorithmTests(unittest.TestCase):
    def test_best_fitness_history_never_decreases(self):
        scores = iter((1.0, 0.0, -22.0, -23.0))
        algorithm = GeneticAlgorithm(
            population_size=2,
            elite_count=1,
            tournament_size=1,
            mutation_rate=0.0,
            mutation_scale=0.0,
            crossover_rate=0.0,
            seed=1,
            fitness_function=lambda world: next(scores),
        )

        algorithm.run(generations=2, steps=0)

        best_history = [entry["best_fitness"] for entry in algorithm.history]
        generation_history = [
            entry["generation_best_fitness"] for entry in algorithm.history
        ]
        self.assertEqual(best_history, [1.0, 1.0])
        self.assertEqual(generation_history, [1.0, -22.0])


if __name__ == "__main__":
    unittest.main()