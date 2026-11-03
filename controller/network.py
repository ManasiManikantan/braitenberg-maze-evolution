"""A small feed-forward neural network for a two-sensor robot controller."""

import numpy as np


class NeuralNetwork:
    """Map left/right sensor readings to left/right motor commands.

    The network has one tanh hidden layer and sigmoid outputs, so motor
    commands are in the range [0, 1].
    """

    def __init__(self, hidden_size=4, seed=None):
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")

        rng = np.random.default_rng(seed)
        self.weights_input_hidden = rng.normal(0.0, 1.0, (2, hidden_size))
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = rng.normal(0.0, 1.0, (hidden_size, 2))
        self.bias_output = np.zeros(2)

    def forward(self, sensors):
        """Return motor commands for ``(left_sensor, right_sensor)``."""
        inputs = np.asarray(sensors, dtype=float)
        if inputs.shape != (2,):
            raise ValueError("sensors must contain exactly two values")

        hidden = np.tanh(inputs @ self.weights_input_hidden + self.bias_hidden)
        logits = hidden @ self.weights_hidden_output + self.bias_output
        logits = np.clip(logits, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-logits))

    __call__ = forward
