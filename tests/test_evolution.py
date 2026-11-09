
from evolution.genetic_algorithm import GeneticAlgorithm

# Create a small population
ga = GeneticAlgorithm(
    population_size=10,
    hidden_size=4,
    mutation_rate=0.1,
    mutation_scale=0.1,
    seed=42
)

# Evolve for 5 generations
best = ga.run(
    generations=5,
    steps=200
)

print("\nEVOLUTION RESULTS")
print("----------------------")
print("Best fitness:", best.fitness)
print("Best genome:", best.genome)
print("Generations:", len(ga.history))

for result in ga.history:
    print(
        "Generation:", result["generation"],
        "Best so far:", round(result["best_fitness"], 3),
        "Generation best:", round(result["generation_best_fitness"], 3),
        "Average:", round(result["mean_fitness"], 3)
    )

print("\nBEST ROBOT PERFORMANCE")
print("----------------------")

for key, value in best.metrics.items():
    print(f"{key}: {value}")