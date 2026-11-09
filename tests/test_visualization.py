
from sim.robot import Vehicle
from sim.world import World
from sim.visualization import animate_world

# Create a robot with no random movement noise
robot = Vehicle(noise_stdev=0)

# Create the simulation
world = World(vehicle=robot)

# Animate the robot
animation = animate_world(
    world,
    steps=500,
    interval=30
)
