
from controller.network import NeuralNetwork

# Create a neural network
network = NeuralNetwork(hidden_size=4, seed=42)

# Simulate left, forward, and right sensor readings
sensors = [0.8, 0.5, 0.2]

# Generate motor commands
motors = network.forward(sensors)

print("Sensor inputs:", sensors)
print("Motor outputs:", motors)

assert len(motors) == 2
assert all(-1 <= motor <= 1 for motor in motors)

network.weights_hidden_output[:] = 0
network.bias_output[:] = [-2, 2]
motors = network.forward(sensors)
assert motors[0] < 0 < motors[1]

print("Neural network test passed!")
