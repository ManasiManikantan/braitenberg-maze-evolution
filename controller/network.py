"""A small feed-forward neural network for a three-sensor robot controller."""

import numpy as np


class NeuralNetwork:
    """Map left/front/right sensor readings to left/right motor commands.

    The network has one tanh hidden layer and tanh outputs, so motor
    commands are in the range [-1, 1] and can drive the robot in reverse.
    """

    def __init__(self, hidden_size=4, seed=None):
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")

        rng = np.random.default_rng(seed)
        self.weights_input_hidden = rng.normal(0.0, 1.0, (3, hidden_size))
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = rng.normal(0.0, 1.0, (hidden_size, 2))
        self.bias_output = np.zeros(2)

    def forward(self, sensors):
        """Return motor commands for ``(left_sensor, front_sensor, right_sensor)``."""
        inputs = np.asarray(sensors, dtype=float)
        if inputs.shape != (3,):
            raise ValueError("sensors must contain exactly three values")

        hidden = np.tanh(inputs @ self.weights_input_hidden + self.bias_hidden)
        logits = hidden @ self.weights_hidden_output + self.bias_output
        return np.tanh(logits)

    __call__ = forward
