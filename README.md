# Braitenberg Maze Evolution

This project aims to evolve neural-network controllers for a Braitenberg-style
differential-drive robot and study whether distinct maze-navigation strategies
emerge. The long-term goal is to support transfer from a 2D simulation to a
Pololu Romi robot.

## Requirements

- Python 3
- Packages listed in [`requirements.txt`](requirements.txt)

Install the dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Neural-network controller

[`controller/network.py`](controller/network.py) defines a small feed-forward
network with two inputs (left and right sensor readings), one configurable
hidden layer, and two outputs (left and right motor commands). The hidden layer
and outputs use `tanh`; motor commands are in the range `[-1, 1]`, where
negative values drive the robot in reverse.

```python
from controller.network import NeuralNetwork

controller = NeuralNetwork(hidden_size=4, seed=7)
left_motor, right_motor = controller.forward((0.2, 0.8))
```

The optional `seed` makes the initial weights reproducible. Calling the
controller directly is equivalent to calling `forward`:

```python
motor_commands = controller((left_sensor, right_sensor))
```

This network initializes random weights; the genetic algorithm below evolves
those parameters.

## Maze simulation

`sim/maze.py` provides a centered rectangular maze with boundary and interior
walls plus rectangular obstacles. `Vehicle.sense(maze)` casts two forward-facing
rays, up to `sensor_range` maze units. `left_distance` and `right_distance` hold
the measured distances; the controller inputs are normalized proximity values
from 0 (no obstacle in range) to 1 (obstacle at the sensor).

`World.step()` runs one sense-think-move cycle and prevents the vehicle body
from crossing walls or obstacles:

```python
from sim.world import World

world = World()
world.run(1000)
vehicle = world.vehicle
```

Pass a custom `Maze` or `Vehicle` to `World` to change the layout or controller.
When both forward rays are blocked, the vehicle scans a 180-degree forward arc,
aligns with the heading that saw the greatest clearance, and then moves toward
that opening; it does not select a route directly behind itself.

## Maze visualization

`sim/visualization.py` animates the walls and obstacles, goal, vehicle pose,
forward sensor rays, and movement trail:

```python
from sim.visualization import animate_world
from sim.world import World

world = World()
animation = animate_world(world, steps=1000, interval=40)
```

The animator displays the resulting pose and sensor geometry for both forward
and reverse motion. A fitness-over-generations plot can be added separately.

## Fitness and MAP-Elites

`evolution/fitness.py` scores a completed run using progress toward the maze
goal, safe clearance from walls, collisions, moving
steps, stuck status, and goal completion. Collision costs increase after
repeated impacts; useful path length and goal approach are rewarded, while
movement away from the goal and getting stuck are penalized. Reaching the goal
has a strong default bonus:

```python
from evolution.fitness import evaluate_fitness
from sim.world import World

world = World()
world.run(1000)
result = evaluate_fitness(world)
score = result["score"]
descriptor = result["behavior_descriptor"]
```

The descriptor is `(coverage, normalized_goal_progress)`. A MAP-Elites archive
can bin those values and retain the highest-scoring controller in each cell,
preserving diverse maze behaviors instead of only one overall winner. The
archive is not implemented yet.

## Genetic algorithm

`evolution/genetic_algorithm.py` evolves a fixed-topology controller population
with tournament selection, single-point crossover, Gaussian mutation, and
elitist generational replacement:

```python
from evolution.genetic_algorithm import GeneticAlgorithm

algorithm = GeneticAlgorithm(population_size=50, seed=7)
best = algorithm.run(generations=20, steps=1000)
best_controller = algorithm.decode_genome(best.genome)
```

Pass `world_factory(controller)` to `run()` to evaluate each controller in a
custom maze or episode setup. `algorithm.history` records best and mean fitness
for each evaluated generation. An optional `progress_callback(generation,
completed, total, fitness)` receives an update after each controller finishes.
`best_fitness` is best-so-far and cannot decrease; `generation_best_fitness`
shows the best score in that generation. Default GA evaluation disables motion
noise for consistent comparisons; set `evaluation_noise_stdev` to a positive
value when noisy evaluation is desired.

## Project layout

| Path | Description |
| --- | --- |
| `controller/network.py` | Two-sensor, two-motor feed-forward neural network |
| `sim/robot.py` | Robot, distance sensors, and differential-drive motion |
| `sim/maze.py` | Maze walls, obstacles, raycasting, and collision checks |
| `sim/world.py` | Sense-think-move simulation loop |
| `sim/visualization.py` | Live maze and robot animation |
| `evolution/fitness.py` | Maze-run scoring and MAP-Elites behavior descriptors |
| `evolution/genetic_algorithm.py` | Generational neural-controller evolution |

The MAP-Elites archive is not implemented yet.
