"""A generational genetic algorithm for neural-network robot controllers."""

import numpy as np

from controller.network import DEFAULT_OUTPUT_BIAS, NeuralNetwork
from evolution.fitness import evaluate_outcome_fitness, evaluate_shaped_fitness
from sim.robot import DEFAULT_ROBOT_NOISE, Vehicle
from sim.world import World


class Individual:
	"""A controller genome and its most recent evaluation."""

	def __init__(self, genome):
		self.genome = np.asarray(genome, dtype=float).copy()
		self.fitness = None
		self.metrics = None


class GeneticAlgorithm:
	"""Evolve fixed-topology ``NeuralNetwork`` controllers.

	The genome concatenates input-hidden weights, hidden biases,
	hidden-output weights, and output biases. Fitness is evaluated by creating
	a fresh ``World`` for each controller, using ``world_factory(controller)``
	when a custom maze or episode setup is needed.

	With ``network_only=True`` (the default) the default world disables the
	hand-coded navigation heuristics, so the evolved network is the only
	controller. Set it to ``False`` to evolve inside the rule-based layer.
	Unless ``fitness_function`` is given, network-only runs are scored with
	``evaluate_outcome_fitness`` and rule-based runs with
	``evaluate_shaped_fitness``.

	The default world applies ``robot_noise`` (``DEFAULT_ROBOT_NOISE`` unless
	given; pass ``{}`` to disable): sensor, per-wheel motor, fixed wheel
	mismatch, and start pose noise, with a fresh reproducible noise seed for
	every evaluation.
	"""

	def __init__(
		self,
		population_size=50,
		hidden_size=4,
		mutation_rate=0.1,
		mutation_scale=0.1,
		crossover_rate=0.8,
		elite_count=2,
		tournament_size=3,
		seed=None,
		fitness_function=None,
		evaluation_noise_stdev=0.0,
		network_only=True,
		initial_output_bias=DEFAULT_OUTPUT_BIAS,
		robot_noise=None,
	):
		if population_size < 2:
			raise ValueError("population_size must be at least 2")
		if hidden_size < 1:
			raise ValueError("hidden_size must be at least 1")
		if not 0 <= mutation_rate <= 1:
			raise ValueError("mutation_rate must be between 0 and 1")
		if mutation_scale < 0:
			raise ValueError("mutation_scale must be non-negative")
		if not 0 <= crossover_rate <= 1:
			raise ValueError("crossover_rate must be between 0 and 1")
		if not 0 <= elite_count < population_size:
			raise ValueError("elite_count must be in [0, population_size)")
		if not 1 <= tournament_size <= population_size:
			raise ValueError("tournament_size must be between 1 and population_size")
		if evaluation_noise_stdev < 0:
			raise ValueError("evaluation_noise_stdev must be non-negative")

		self.population_size = population_size
		self.hidden_size = hidden_size
		self.mutation_rate = mutation_rate
		self.current_mutation_rate = mutation_rate
		self.stagnant_generations = 0
		self.mutation_scale = mutation_scale
		self.crossover_rate = crossover_rate
		self.elite_count = elite_count
		self.tournament_size = tournament_size
		self.evaluation_noise_stdev = evaluation_noise_stdev
		self.network_only = network_only
		self.initial_output_bias = float(initial_output_bias)
		self.robot_noise = dict(DEFAULT_ROBOT_NOISE if robot_noise is None else robot_noise)
		self.rng = np.random.default_rng(seed)
		# Separate stream so per-evaluation noise seeds do not perturb the
		# GA's own selection/mutation random numbers.
		self._noise_seed_rng = np.random.default_rng(None if seed is None else [seed, 1])
		if fitness_function is None:
			fitness_function = (
				evaluate_outcome_fitness if network_only else evaluate_shaped_fitness
			)
		self.fitness_function = fitness_function
		self.genome_size = 6 * hidden_size + 2
		self.population = self.initialize_population()
		self.generation = 0
		self.best_individual = None
		self.history = []

	def _random_genome(self):
		input_hidden = self.rng.normal(0.0, 1.0, (3, self.hidden_size))
		bias_hidden = np.zeros(self.hidden_size)
		hidden_output = self.rng.normal(0.0, 1.0, (self.hidden_size, 2))
		bias_output = np.full(2, self.initial_output_bias)
		return np.concatenate((
			input_hidden.ravel(),
			bias_hidden,
			hidden_output.ravel(),
			bias_output,
		))

	def initialize_population(self):
		"""Create a fresh population of independently initialized genomes."""
		return [Individual(self._random_genome()) for _ in range(self.population_size)]

	def encode_network(self, network):
		"""Flatten a compatible neural network into this algorithm's genome format."""
		if network.weights_input_hidden.shape != (3, self.hidden_size):
			raise ValueError("network hidden size does not match the genetic algorithm")
		if network.weights_hidden_output.shape != (self.hidden_size, 2):
			raise ValueError("network hidden size does not match the genetic algorithm")
		return np.concatenate((
			network.weights_input_hidden.ravel(),
			network.bias_hidden,
			network.weights_hidden_output.ravel(),
			network.bias_output,
		))

	def decode_genome(self, genome):
		"""Build a ``NeuralNetwork`` with parameters copied from ``genome``."""
		genome = np.asarray(genome, dtype=float)
		if genome.shape != (self.genome_size,):
			raise ValueError(f"genome must contain exactly {self.genome_size} values")

		network = NeuralNetwork(hidden_size=self.hidden_size, seed=0)
		cursor = 0
		input_hidden_size = 3 * self.hidden_size
		network.weights_input_hidden = genome[cursor:cursor + input_hidden_size].reshape(
			3, self.hidden_size
		).copy()
		cursor += input_hidden_size
		network.bias_hidden = genome[cursor:cursor + self.hidden_size].copy()
		cursor += self.hidden_size
		hidden_output_size = 2 * self.hidden_size
		network.weights_hidden_output = genome[cursor:cursor + hidden_output_size].reshape(
			self.hidden_size, 2
		).copy()
		cursor += hidden_output_size
		network.bias_output = genome[cursor:cursor + 2].copy()
		return network

	def select_parent(self):
		"""Select the fittest individual from a random tournament."""
		if any(individual.fitness is None for individual in self.population):
			raise RuntimeError("evaluate the population before selecting parents")
		indices = self.rng.choice(
			len(self.population), size=self.tournament_size, replace=False
		)
		return max((self.population[index] for index in indices), key=lambda item: item.fitness)

	def crossover(self, first, second):
		"""Create two children using single-point crossover."""
		first = np.asarray(first, dtype=float)
		second = np.asarray(second, dtype=float)
		if first.shape != (self.genome_size,) or second.shape != (self.genome_size,):
			raise ValueError("parent genomes have an invalid size")
		if self.genome_size < 2 or self.rng.random() >= self.crossover_rate:
			return first.copy(), second.copy()

		crossover_point = self.rng.integers(1, self.genome_size)
		child_one = np.concatenate((first[:crossover_point], second[crossover_point:]))
		child_two = np.concatenate((second[:crossover_point], first[crossover_point:]))
		return child_one, child_two

	def mutate(self, genome):
		"""Apply independent Gaussian mutation to selected genes."""
		mutated = np.asarray(genome, dtype=float).copy()
		if mutated.shape != (self.genome_size,):
			raise ValueError("genome has an invalid size")
		mask = self.rng.random(self.genome_size) < self.current_mutation_rate
		mutated[mask] += self.rng.normal(0.0, self.mutation_scale, mask.sum())
		return mutated

	def _default_world_factory(self, controller):
		return World(vehicle=Vehicle(
			controller=controller,
			noise_stdev=self.evaluation_noise_stdev,
			navigation_heuristics=not self.network_only,
			noise_seed=int(self._noise_seed_rng.integers(2**32)),
			**self.robot_noise,
		))

	def evaluate_population(self, world_factory=None, steps=1000, progress_callback=None):
		"""Run and score every individual, returning the generation's best."""
		if steps < 0:
			raise ValueError("steps must be non-negative")
		factory = self._default_world_factory if world_factory is None else world_factory
		for index, individual in enumerate(self.population):
			controller = self.decode_genome(individual.genome)
			world = factory(controller)
			world.run(steps)
			result = self.fitness_function(world)
			if isinstance(result, dict):
				fitness = float(result["score"])
				individual.metrics = result
			else:
				fitness = float(result)
				individual.metrics = {"score": fitness}
			if not np.isfinite(fitness):
				raise ValueError("fitness values must be finite")
			individual.fitness = fitness
			if self.best_individual is None or fitness > self.best_individual.fitness:
				self.best_individual = Individual(individual.genome)
				self.best_individual.fitness = fitness
				self.best_individual.metrics = individual.metrics
			if progress_callback is not None:
				progress_callback(
					self.generation,
					index + 1,
					len(self.population),
					self.best_individual.fitness,
				)

		best = max(self.population, key=lambda item: item.fitness)
		if self.history and best.fitness <= self.history[-1]["generation_best_fitness"]:
			self.stagnant_generations += 1
		else:
			self.stagnant_generations = 0
		self.current_mutation_rate = min(
			0.5,
			self.mutation_rate * (1.0 + 0.5 * self.stagnant_generations),
		)
		self.history.append({
			"generation": self.generation,
			"best_fitness": self.best_individual.fitness,
			"generation_best_fitness": best.fitness,
			"mean_fitness": float(np.mean([item.fitness for item in self.population])),
			"stagnant_generations": self.stagnant_generations,
			"mutation_rate": self.current_mutation_rate,
		})
		return best

	def next_generation(self):
		"""Replace the population with elites and mutated offspring."""
		if any(individual.fitness is None for individual in self.population):
			raise RuntimeError("evaluate the population before creating a generation")

		ranked = sorted(self.population, key=lambda item: item.fitness, reverse=True)
		new_population = [Individual(item.genome) for item in ranked[:self.elite_count]]
		while len(new_population) < self.population_size:
			parent_one = self.select_parent()
			parent_two = self.select_parent()
			children = self.crossover(parent_one.genome, parent_two.genome)
			for child in children:
				new_population.append(Individual(self.mutate(child)))
				if len(new_population) == self.population_size:
					break

		self.population = new_population
		self.generation += 1
		return self.population

	def run(self, generations, steps=1000, world_factory=None, progress_callback=None):
		"""Evaluate successive generations and return the best individual found."""
		if generations < 1:
			raise ValueError("generations must be at least 1")
		for generation_index in range(generations):
			self.evaluate_population(
				world_factory=world_factory,
				steps=steps,
				progress_callback=progress_callback,
			)
			if generation_index < generations - 1:
				self.next_generation()
		return self.best_individual