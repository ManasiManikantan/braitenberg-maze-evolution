"""A small feed-forward neural network for a three-sensor robot controller."""

import numpy as np

# Initial output bias for both motors. With all sensors at 0 (nothing in
# range) the hidden layer outputs 0, so the motors read tanh(bias); a zero
# bias would leave a fresh random network sitting still in open space.
# tanh(0.5) ~= 0.46 gives a moderate forward drive that evolution can retune.
DEFAULT_OUTPUT_BIAS = 0.5


class NeuralNetwork:
    """Map sensor readings to left/right motor commands.

    By default the inputs are the left/front/right distance sensors; with
    ``input_size=5`` the left and right light sensors follow them. The network
    has one tanh hidden layer and tanh outputs, so motor commands are in the
    range [-1, 1] and can drive the robot in reverse. Output biases start at
    ``output_bias`` so untrained networks move forward.
    """

    def __init__(self, hidden_size=4, seed=None, output_bias=DEFAULT_OUTPUT_BIAS, input_size=3):
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if input_size < 1:
            raise ValueError("input_size must be at least 1")

        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.weights_input_hidden = rng.normal(0.0, 1.0, (input_size, hidden_size))
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = rng.normal(0.0, 1.0, (hidden_size, 2))
        self.bias_output = np.full(2, float(output_bias))

    def forward(self, sensors):
        """Return motor commands for the sensor readings (``input_size`` values)."""
        inputs = np.asarray(sensors, dtype=float)
        if inputs.shape != (self.input_size,):
            raise ValueError(f"sensors must contain exactly {self.input_size} values")

        hidden = np.tanh(inputs @ self.weights_input_hidden + self.bias_hidden)
        logits = hidden @ self.weights_hidden_output + self.bias_output
        return np.tanh(logits)

    __call__ = forward
