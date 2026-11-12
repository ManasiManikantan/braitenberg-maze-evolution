import unittest

from evolution.genetic_algorithm import GeneticAlgorithm


class GeneticAlgorithmTests(unittest.TestCase):
    def test_mutation_rate_increases_after_stagnation(self):
        algorithm = GeneticAlgorithm(
            population_size=2,
            elite_count=1,
            tournament_size=1,
            mutation_rate=0.1,
            mutation_scale=0.1,
            crossover_rate=0.0,
            seed=2,
            fitness_function=lambda world: 1.0,
        )

        algorithm.run(generations=4, steps=0)

        self.assertEqual(algorithm.history[0]["mutation_rate"], 0.1)
        self.assertEqual(algorithm.history[-1]["stagnant_generations"], 3)
        self.assertAlmostEqual(algorithm.history[-1]["mutation_rate"], 0.25)

    def test_genome_encodes_three_sensor_inputs(self):
        algorithm = GeneticAlgorithm(population_size=2, hidden_size=2, seed=1)
        network = algorithm.decode_genome(algorithm.population[0].genome)

        self.assertEqual(algorithm.genome_size, 14)
        self.assertEqual(network.weights_input_hidden.shape, (3, 2))

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

    def test_default_world_is_network_only(self):
        algorithm = GeneticAlgorithm(population_size=4, seed=1)
        controller = algorithm.decode_genome(algorithm.population[0].genome)
        world = algorithm._default_world_factory(controller)
        self.assertFalse(world.vehicle.navigation_heuristics)

        with_rules = GeneticAlgorithm(population_size=4, seed=1, network_only=False)
        world = with_rules._default_world_factory(controller)
        self.assertTrue(world.vehicle.navigation_heuristics)


if __name__ == "__main__":
    unittest.main()