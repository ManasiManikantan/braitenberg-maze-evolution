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
network with three inputs (left, forward, and right sensor readings), one configurable
hidden layer, and two outputs (left and right motor commands). The hidden layer
and outputs use `tanh`; motor commands are in the range `[-1, 1]`, where
negative values drive the robot in reverse.

```python
from controller.network import NeuralNetwork

controller = NeuralNetwork(hidden_size=4, seed=7)
left_motor, right_motor = controller.forward((0.2, 0.5, 0.8))
```

The optional `seed` makes the initial weights reproducible. Calling the
controller directly is equivalent to calling `forward`:

```python
motor_commands = controller((left_sensor, front_sensor, right_sensor))
```

This network initializes random weights; the genetic algorithm below evolves
those parameters.

## Maze simulation

`sim/maze.py` provides a centered rectangular maze with boundary and interior
walls plus rectangular obstacles. `Maze(difficulty=1)` through
`Maze(difficulty=4)` select the four built-in courses.
`Vehicle.sense(maze)` casts three parallel, forward-facing rays: one from the
front and two from the robot's left/right body edges. Each reaches up to
`sensor_range` maze units. Their distances are passed to the controller in
left/front/right order. `left_distance`, `front_distance`, and `right_distance` hold
the measured distances. Wall hits use a smaller safety margin than rectangular
obstacles, so the robot can pass close to walls without cutting obstacle gaps
too tightly. Controller inputs are normalized proximity values
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
When the goal is unobstructed and within `goal_seek_distance` (4 maze units by
default), the robot steers toward it; at greater distances it keeps navigating
until the goal is closer. Walls and obstacles keep the goal hidden until a path
opens.
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
region (when the robot body overlaps it; 2 units center-to-center by default)
earns a 5-point completion bonus. Fitness reports center error and overlap
margin separately:

```python
from evolution.fitness import evaluate_fitness
from sim.world import World

world = World()
world.run(1000)
result = evaluate_fitness(world)
score = result["score"]
descriptor = result["behavior_descriptor"]
```

The descriptor is `(coverage, normalized_goal_progress)`. `evolution/map_elites.py`
bins those values into a grid and keeps the highest-scoring controller in each
cell, preserving diverse maze behaviors instead of only one overall winner:

```bash
python -m evolution.map_elites --course 1 --evaluations 1500
```

The search seeds the archive with random controllers, then repeatedly picks
elites from the archive, applies crossover and mutation, and inserts the child
if it beats its cell's current elite. Archive score is the fitness score
(clipped at -100) plus 100 for reaching the goal. By default, the run creates
the `results` and `test_output` folders in the current working directory and writes:

- `test_output/map_elites_archive.png`: a heatmap of best fitness per behavior cell (gold
  stars mark cells whose elite reaches the goal) beside the paths of the 12
  highest-fitness elites drawn on the maze.
- `results/map_elites_archive.npz`: every elite's genome, cell, descriptor, and fitness
  so controllers can be replayed.
- `results/map_elites_evaluations.csv`: every controller evaluated (not just the
  winners), with its cell, descriptor, fitness, goal success, collisions,
  completion steps, path length, whether it entered the archive, and its genome
  as `gene_0`..`gene_25`. This is the full dataset for later training or analysis.
- `results/map_elites_elites.csv`: the same columns for just the final archive.

Later runs overwrite those default files. Override individual paths with
`--output`, `--save`, `--evaluations-csv`, and `--elites-csv`; parent folders
for custom paths must already exist. Visual test figures also use `test_output`.

To start a new search with saved elites, pass
`--seed-archive results/personality-5000/archive.npz`. Every saved genome is
re-evaluated with the new run's course, step limit, and other settings before
being inserted. Seed evaluations count toward `--evaluations`; `--initial`
additional random controllers are then evaluated if the budget permits, and
the remaining evaluations generate mutated offspring. The budget must be at
least the number of saved genomes. The new archive records `seed_count`, and
the seed evaluations appear first in the evaluation CSV. This is a new seeded
search, not an exact continuation of the old random state or evaluation history.
Use different output paths to preserve the seed archive and earlier results.

Useful options: `--course 1-4`, `--bins 10 10`, `--coverage-range 0.03 0.2`,
`--progress-max 0.85` (a robot stops once it overlaps the goal, so progress
never reaches 1.0), `--neural-only` (disable the hard-coded heuristics so the
archive reflects only the network), `--show` (open the figure window), and
`--seed`.

### Exploring robot personalities

Use the optional personality archive to preserve combinations of direct or
exploratory routes and collision-free or collision-tolerant behavior:

```bash
python -m evolution.map_elites --personality --course 1 --evaluations 1500 --show
```

Each run can use up to 6,000 simulation cycles instead of the usual 3,000,
allowing detours more time to finish. Override this with `--steps`; the robot
still stops as soon as it reaches the goal.

The horizontal axis measures **directness**: net progress toward the goal
divided by distance traveled, clamped to 0–1 (zero when the robot does not
move). The vertical axis counts actual simulated collision events; the top row
groups counts at or above the last row's number. Collisions remain physically
blocked by the simulator. In each cell, reaching the goal takes priority, then
fitness selects the best controller. Personality-mode fitness removes the
ordinary collision penalties so collision-heavy controllers can compete; other
fitness terms and the goal bonus remain in place. In addition, explorer-labeled
runs get 2 points per unique maze cell visited, on top of the existing coverage
reward. The base fitness still penalizes each re-entry into a previously visited
cell by 0.5 points; there is no extra explorer-specific revisit penalty. The
new-area bonus is applied after the score floor and goal bonus. Goal-reaching
runs still take priority in each cell, and the normal mode is unchanged.

The cells used for these metrics are 1-by-1 maze units. Staying in the same
cell for consecutive steps does not count as a revisit; leaving and coming
back does. This approximates path overlap rather than measuring exact line
crossings. The CSV and NPZ outputs include the unique-cell count, revisit count,
and exploration adjustment so you can compare runs. The gallery also shows
the cell and revisit counts for each representative.

The gallery shows only the best available path for each distinct observed label:

- **direct**: reached the goal with directness at least 0.4.
- **explorer**: not direct, but visited enough distinct maze cells to reach the
  coverage threshold of 0.08 (8% of the maze's grid cells).
- **limited exploration**: neither of the above; this can include a robot that
  hardly moved.
- Each is combined with **collision-free** (zero impacts) or
  **collision-tolerant** (one or more impacts). Unfinished runs are marked.

These are labels of observed behavior, not evidence of intent or guarantees
about future runs. A collision-free run is not necessarily cautious, and a
collision-tolerant run is not guaranteed to reach the goal. The search may not
find every combination; these fixed thresholds are rough guides and may need
adjustment for different courses. The hard-coded avoidance and goal-seeking heuristics can
also influence the paths; `--neural-only` disables them for comparison.

The CSV and NPZ outputs include directness, coverage, goal progress, collision
counts, and genomes. NPZ descriptors are `(directness, collisions)` in
personality mode and `(coverage, progress)` otherwise. Use separate output paths
to keep both runs; the default filenames are the same in either mode.

## Genetic algorithm

`evolution/genetic_algorithm.py` evolves a fixed-topology controller population
with tournament selection, single-point crossover, Gaussian mutation, and
elitist generational replacement:

```python
from evolution.genetic_algorithm import GeneticAlgorithm

algorithm = GeneticAlgorithm(population_size=50, seed=7)
best = algorithm.run(generations=20, steps=2000)
best_controller = algorithm.decode_genome(best.genome)
```

Pass `world_factory(controller)` to `run()` to evaluate each controller in a
custom maze or episode setup. `algorithm.history` records best and mean fitness
for each evaluated generation. An optional `progress_callback(generation,
completed, total, fitness)` receives an update after each controller finishes.
`best_fitness` is best-so-far and cannot decrease; `generation_best_fitness`
shows the best score in that generation. Default GA evaluation disables motion
noise for consistent comparisons; set `evaluation_noise_stdev` to a positive
value when noisy evaluation is desired. If generation-best fitness stagnates,
the mutation rate increases by 50% per generation up to 0.5, then resets after
an improvement.
Each fitness result includes `score_components`, so progress rewards, safety,
collision costs, movement-time costs, and the goal bonus can be inspected
separately instead of interpreting the total score alone. It also reports
`coverage`, `unique_cells_visited`, and `path_revisits`; revisiting old grid
cells is penalized while exploring new cells is rewarded. During a run, the
world requests a fresh forward scan when the robot re-enters a visited grid
cell or makes no meaningful goal progress for 80 steps (configurable via
`World.stagnation_limit` and `World.progress_tolerance`).

## Reproducible controller study

Run the study from the project root:

```bash
python tests/test_evolved_robot.py
```

By default, the study trains three independent genetic-algorithm runs (10
generations, population 30, and 5,000 steps per course). Each run evaluates a
randomly initialized neural controller on all four training courses and saves
those baseline measurements before evolution begins. The evolved controller
is then evaluated on the same courses and two held-out layouts, with seeded
movement noise (standard deviation 0.05).

The study also evaluates the random and evolved neural controllers with the
built-in goal-seeking and obstacle-avoidance heuristics disabled, plus a
heuristic-only robot without a neural controller. This ablation distinguishes
the neural controller's contribution from existing hand-coded behavior. CSV
rows include goal success, collision count, completion steps (blank when the
goal was not reached), path length, fitness, course, controller, and seed. The
runner reports aggregate success rates and whether each reaches the 80%
goal-success target. Results are saved to `experiment_results.csv`. The study
also saves each seed's random and evolved controller genomes to
`experiment_genomes.npz` (keys like `evolved_seed0`) and the per-generation
training curve to `experiment_training_history.csv`.

Workload and output can be adjusted:

```bash
python tests/test_evolved_robot.py --seeds 3 --generations 10 --steps 5000 --noise-stdev 0.05 --output experiment_results.csv
```

## Project layout

| Path | Description |
| --- | --- |
| `controller/network.py` | Three-sensor, two-motor feed-forward neural network |
| `sim/robot.py` | Robot, distance sensors, and differential-drive motion |
| `sim/maze.py` | Maze walls, obstacles, raycasting, and collision checks |
| `sim/world.py` | Sense-think-move simulation loop |
| `sim/visualization.py` | Live maze and robot animation |
| `evolution/fitness.py` | Maze-run scoring and MAP-Elites behavior descriptors |
| `evolution/genetic_algorithm.py` | Generational neural-controller evolution |
| `evolution/experiment.py` | Reproducible baseline, validation, and ablation study |
| `evolution/map_elites.py` | MAP-Elites archive search and visualization |
