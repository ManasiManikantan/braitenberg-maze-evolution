
from controller.network import NeuralNetwork

# Create a neural network
network = NeuralNetwork(hidden_size=4, seed=42)

# Simulate two sensor readings
sensors = [0.8, 0.2]

# Generate motor commands
motors = network.forward(sensors)

print("Sensor inputs:", sensors)
print("Motor outputs:", motors)

assert len(motors) == 2
assert all(0 <= motor <= 1 for motor in motors)

print("Neural network test passed!")
