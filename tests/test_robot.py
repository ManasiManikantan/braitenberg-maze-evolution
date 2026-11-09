
from sim.robot import Vehicle
from sim.world import World

# Create a robot with no random movement noise
robot = Vehicle(noise_stdev=0)

# Put it inside the maze
world = World(vehicle=robot)

print("Starting position:", robot.x_pos, robot.y_pos)

# Run 100 simulation steps
for i in range(100):
    world.step()

print("Ending position:", robot.x_pos, robot.y_pos)
print("Left sensor:", robot.left_sensor)
print("Right sensor:", robot.right_sensor)
print("Left motor:", robot.left_motor)
print("Right motor:", robot.right_motor)
print("Collisions:", world.collisions)
print("Steps:", world.steps)
