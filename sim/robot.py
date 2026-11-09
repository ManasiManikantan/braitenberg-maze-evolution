"""
Braitenberg vehicle simulation.

Defines the core building blocks for a maze-navigation robot:
  - Point      : base class for any 2-D object with a position.
    - Vehicle    : a two-wheeled robot with two distance sensors and a differential drive.
    - Light      : a legacy stationary point light source.
"""

import numpy as np


def euclidean_distance(point1, point2):
    """Return the Euclidean distance between two Point objects."""
    return np.sqrt((point1.x_pos - point2.x_pos)**2 + (point1.y_pos - point2.y_pos)**2)


class Point:
    """A 2-D object with an (x, y) position."""

    def __init__(self, x_pos=0.0, y_pos=0.0):
        self.x_pos = x_pos
        self.y_pos = y_pos


class Vehicle(Point):
    """A differential-drive vehicle with two forward-facing distance sensors.

    The vehicle body is modelled as a circle. Two distance sensors are mounted
    at the front of the body and angled symmetrically around its heading.

    Args:
        angle_offset: Angle between each sensor ray and the forward axis.
        turn_gain:    How strongly the sensor difference steers the vehicle.
                      Larger values produce sharper turns.
        noise_stdev:  Standard deviation of Gaussian noise added to the
                      orientation at each step.  Zero means no noise.
        sensor_range: Maximum sensor distance in maze units.
        wiring:       Sensor-to-motor mapping: "crossed" (default) or "direct".
        controller:   Optional controller with a two-input, two-output API.
        distance:     Legacy alias for sensor_range.
    """

    def __init__(self, angle_offset=np.pi/6, turn_gain=0.1, noise_stdev=0.1,
                 sensor_range=10, wiring="crossed", controller=None, distance=None):
        super().__init__()
        # Every Vehicle starts at the origin facing along +x. The only thing
        # that varies between repetitions is the random noise.
        self.orientation = 0.0          # heading angle (radians); starts pointing along +x
        self.velocity = 0.0             # forward speed (arbitrary units)
        self.radius = 1.0               # body radius; also the sensor arm length

        self.left_sensor  = 0.0        # current left sensor activation  [0, 1]
        self.right_sensor = 0.0        # current right sensor activation [0, 1]
        self.left_motor   = 0.0        # left motor command
        self.right_motor  = 0.0        # right motor command

        if distance is not None:
            sensor_range = distance
        if sensor_range <= 0:
            raise ValueError("sensor_range must be positive")
        self.sensor_range = sensor_range
        self.left_distance = sensor_range
        self.right_distance = sensor_range

        self.turn_gain   = turn_gain
        self.vel_gain    = 1 / 50       # converts average motor output to forward speed
        self.noise_stdev = noise_stdev
        self.angle_offset = angle_offset
        self.wiring = wiring
        self.controller = controller

        # Sensor positions as Point objects, updated every step
        self.rs = Point()   # right sensor
        self.ls = Point()   # left sensor
        self.update_sensor_pos()

    def update_sensor_pos(self):
        """Recompute the front-left and front-right sensor origins."""
        left_angle = self.orientation + self.angle_offset
        right_angle = self.orientation - self.angle_offset
        self.ls.x_pos = self.x_pos + self.radius * np.cos(left_angle)
        self.ls.y_pos = self.y_pos + self.radius * np.sin(left_angle)
        self.rs.x_pos = self.x_pos + self.radius * np.cos(right_angle)
        self.rs.y_pos = self.y_pos + self.radius * np.sin(right_angle)

    def update_body_pos(self):
        """Advance the body one step along the current heading at the current velocity."""
        self.x_pos += self.velocity * np.cos(self.orientation)
        self.y_pos += self.velocity * np.sin(self.orientation)
        self.update_sensor_pos()

    def sense(self, maze):
        """Read normalized proximity to the nearest wall or obstacle.

        Sensor values range from 0 (nothing detected within range) to 1
        (an obstacle at the sensor origin). Raw distances are also exposed as
        ``left_distance`` and ``right_distance``.

        Args:
            maze: Maze providing a ``raycast`` method.
        """
        left_angle = self.orientation + self.angle_offset
        right_angle = self.orientation - self.angle_offset
        self.left_distance = maze.raycast(
            self.ls,
            (np.cos(left_angle), np.sin(left_angle)),
            self.sensor_range,
        )
        self.right_distance = maze.raycast(
            self.rs,
            (np.cos(right_angle), np.sin(right_angle)),
            self.sensor_range,
        )
        self.left_sensor = 1.0 - self.left_distance / self.sensor_range
        self.right_sensor = 1.0 - self.right_distance / self.sensor_range

    def think(self):
        """Map sensor activations to motor commands using the selected wiring."""
        if self.controller is not None:
            self.left_motor, self.right_motor = self.controller.forward(
                (self.left_sensor, self.right_sensor)
            )
        elif self.wiring == "crossed":
            self.left_motor = self.right_sensor
            self.right_motor = self.left_sensor
        elif self.wiring == "direct":
            self.left_motor = self.left_sensor
            self.right_motor = self.right_sensor
        else:
            raise ValueError("wiring must be 'crossed' or 'direct'")

    def move(self, maze=None):
        """Update orientation and velocity, then advance the body position.

        Differential drive kinematics:
          - Turning rate  = turn_gain × (left_motor − right_motor) + noise
          - Forward speed = vel_gain  × average(left_motor, right_motor)

        Adding noise to the orientation simulates imperfect actuators and
        prevents the vehicle from following a perfectly straight trajectory.
        """
        noise = np.random.normal(0, self.noise_stdev)
        self.orientation += self.turn_gain * (self.left_motor - self.right_motor) + noise
        self.velocity = self.vel_gain * ((self.right_motor + self.left_motor) / 2)
        old_position = (self.x_pos, self.y_pos)
        self.update_body_pos()
        collided = maze is not None and maze.collides(
            (self.x_pos, self.y_pos), self.radius
        )
        if collided:
            self.x_pos, self.y_pos = old_position
            self.update_sensor_pos()
        return collided

    def distance(self, point):
        """Return the Euclidean distance from the vehicle body centre to a point."""
        return euclidean_distance(self, point)


class Light(Point):
    """A stationary point light source.

    Args:
        distance: Distance from the origin at which the light is placed.
                  The light is positioned along the positive x-axis.
    """

    def __init__(self, distance=10.0):
        angle = 0.0   # light is placed along the +x axis
        super().__init__(x_pos=distance * np.cos(angle), y_pos=distance * np.sin(angle))
