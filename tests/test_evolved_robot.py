
from evolution.fitness import evaluate_fitness
from evolution.genetic_algorithm import GeneticAlgorithm
from sim.maze import Maze
from sim.robot import Vehicle
from sim.world import World
from sim.visualization import animate_world
import matplotlib.pyplot as plt

# Evolve a small population
generations = 10
training_steps = 2200
course_difficulties = (1, 2, 3, 4)


def show_progress(generation, completed, total, fitness):
    message = (
        f"Generation {generation + 1}/{generations} | "
        f"controller {completed}/{total} | best so far {fitness:.2f}"
    )
    print(message, end="\r" if completed < total else "\n", flush=True)


class MultiCourseWorld:
    def __init__(self, controller):
        self.courses = [
            (
                difficulty,
                World(
                    maze=Maze(difficulty=difficulty),
                    vehicle=Vehicle(controller=controller, noise_stdev=0),
                ),
            )
            for difficulty in course_difficulties
        ]

    def run(self, steps):
        for _, world in self.courses:
            world.run(steps)


def evaluate_course_set(multi_course_world):
    results = [
        (difficulty, evaluate_fitness(world))
        for difficulty, world in multi_course_world.courses
    ]
    score_components = {
        component: sum(result["score_components"][component] for _, result in results)
        / len(results)
        for component in results[0][1]["score_components"]
    }
    return {
        "score": sum(result["score"] for _, result in results) / len(results),
        "score_components": score_components,
        "courses": [
            {"difficulty": difficulty, **result}
            for difficulty, result in results
        ],
    }


ga = GeneticAlgorithm(
    population_size=30,
    hidden_size=4,
    mutation_rate=0.15,
    mutation_scale=0.2,
    seed=42,
    fitness_function=evaluate_course_set,
)

best = ga.run(
    generations=generations,
    steps=training_steps,
    world_factory=MultiCourseWorld,
    progress_callback=show_progress,
)

print("Best fitness:", best.fitness)
print("Average fitness breakdown across courses:")
for component, value in best.metrics["score_components"].items():
    print(f"  {component}: {value:+.2f}")
for result in best.metrics["courses"]:
    print(
        f"Course {result['difficulty']}: "
        f"reached goal={result['reached_goal']}, "
        f"goal center error={result['goal_center_error']:.3f}, "
        f"goal bonus={result['goal_bonus_awarded']:.2f}"
    )

# Reconstruct the best neural network
controller = ga.decode_genome(best.genome)

# Show the evolved robot on all four courses in separate windows.
animations = [
    animate_world(
        World(
            maze=Maze(difficulty=difficulty),
            vehicle=Vehicle(controller=controller, noise_stdev=0),
        ),
        steps=training_steps,
        interval=30,
        show=False,
    )
    for difficulty in course_difficulties
]
plt.show()
