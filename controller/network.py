"""A small feed-forward neural network for a three-sensor robot controller."""

import numpy as np

# Initial output bias for both motors. With all sensors at 0 (nothing in
# range) the hidden layer outputs 0, so the motors read tanh(bias); a zero
# bias would leave a fresh random network sitting still in open space.
# tanh(0.5) ~= 0.46 gives a moderate forward drive that evolution can retune.
DEFAULT_OUTPUT_BIAS = 0.5


class NeuralNetwork:
    """Map left/front/right sensor readings to left/right motor commands.

    The network has one tanh hidden layer and tanh outputs, so motor
    commands are in the range [-1, 1] and can drive the robot in reverse.
    Output biases start at ``output_bias`` so untrained networks move forward.
    """

    def __init__(self, hidden_size=4, seed=None, output_bias=DEFAULT_OUTPUT_BIAS):
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")

        rng = np.random.default_rng(seed)
        self.weights_input_hidden = rng.normal(0.0, 1.0, (3, hidden_size))
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = rng.normal(0.0, 1.0, (hidden_size, 2))
        self.bias_output = np.full(2, float(output_bias))

    def forward(self, sensors):
        """Return motor commands for ``(left_sensor, front_sensor, right_sensor)``."""
        inputs = np.asarray(sensors, dtype=float)
        if inputs.shape != (3,):
            raise ValueError("sensors must contain exactly three values")

        hidden = np.tanh(inputs @ self.weights_input_hidden + self.bias_hidden)
        logits = hidden @ self.weights_hidden_output + self.bias_output
        return np.tanh(logits)

    __call__ = forward
