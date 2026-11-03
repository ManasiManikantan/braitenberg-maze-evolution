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
uses `tanh`; the outputs use a sigmoid and are in the range `[0, 1]`.

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

This network initializes random weights; training or evolution is not
implemented yet.

## Project layout

| Path | Description |
| --- | --- |
| `controller/network.py` | Two-sensor, two-motor feed-forward neural network |
| `sim/robot.py` | Robot, point, and light simulation primitives |
| `sim/maze.py` | Maze simulation module (not implemented yet) |
| `sim/world.py` | World simulation module (not implemented yet) |
| `evolution/fitness.py` | Fitness evaluation module (not implemented yet) |

There is not yet a complete runnable simulation or evolutionary training
workflow. In particular, `Vehicle.think()` in `sim/robot.py` is not implemented
and raises `NotImplementedError`.
