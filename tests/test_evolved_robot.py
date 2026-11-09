
from evolution.genetic_algorithm import GeneticAlgorithm
from sim.maze import Maze
from sim.robot import Vehicle
from sim.world import World
from sim.visualization import animate_world

# Evolve a small population
generations = 20
training_steps = 3000


def show_progress(generation, completed, total, fitness):
    message = (
        f"Generation {generation + 1}/{generations} | "
        f"controller {completed}/{total} | best so far {fitness:.2f}"
    )
    print(message, end="\r" if completed < total else "\n", flush=True)


def make_challenge_world(controller):
    return World(
        maze=Maze(difficulty=2),
        vehicle=Vehicle(controller=controller, noise_stdev=0),
    )


ga = GeneticAlgorithm(
    population_size=30,
    hidden_size=4,
    mutation_rate=0.15,
    mutation_scale=0.2,
    seed=42
)

best = ga.run(
    generations=generations,
    steps=training_steps,
    world_factory=make_challenge_world,
    progress_callback=show_progress,
)

print("Best fitness:", best.fitness)
print("Reached goal:", best.metrics["reached_goal"])
print("Goal center error:", round(best.metrics["goal_center_error"], 3))
print("Goal bonus:", best.metrics["goal_bonus_awarded"])
print("Fitness breakdown:")
for component, value in best.metrics["score_components"].items():
    print(f"  {component}: {value:+.2f}")

# Reconstruct the best neural network
controller = ga.decode_genome(best.genome)

# Put the evolved controller into a robot
robot = Vehicle(
    controller=controller,
    noise_stdev=0
)

world = World(maze=Maze(difficulty=2), vehicle=robot)

# Watch the evolved robot
animation = animate_world(
    world,
    steps=3000,
    interval=30
)
