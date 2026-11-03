"""
Braitenberg vehicle simulation.

Defines the core building blocks for a light-seeking robot:
  - Point      : base class for any 2-D object with a position.
  - Vehicle    : a two-wheeled robot with two light sensors and a differential drive.
  - Light      : a stationary light source placed at a fixed distance from the origin.

The vehicle's sensor-to-motor wiring is NOT implemented — that's Part 1 of
the assignment (see Vehicle.think). Two schemes are worth trying:
  - "crossed" (contralateral): left sensor -> right motor, right sensor ->
    left motor. This produces light-seeking behavior.
  - "direct" (ipsilateral): left sensor -> left motor, right sensor ->
    right motor. This produces qualitatively different behavior.
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
    """A Braitenberg vehicle with two light sensors and differential drive motors.

    The vehicle body is modelled as a circle of given radius.  Two sensors are
    mounted on the perimeter, symmetrically offset from the forward direction
    by ±angle_offset radians.

    Args:
        angle_offset: Angular separation between each sensor and the vehicle's
                      forward axis (radians).  π/2 places sensors at 90° on
                      each side; smaller values move them closer to the front.
        turn_gain:    How strongly the sensor difference steers the vehicle.
                      Larger values produce sharper turns.
        noise_stdev:  Standard deviation of Gaussian noise added to the
                      orientation at each step.  Zero means no noise.
        distance:     Initial distance to the light source; used to scale the
                      sensor gain so that raw sensor values stay near [0, 1].
        wiring:       Sensor-to-motor wiring scheme, only used if you
                      implement the OPTIONAL advanced part of the
                      assignment (a runtime switch in Vehicle.think).
                      Otherwise unused. Default "crossed".
    """

    def __init__(self, angle_offset=np.pi/2, turn_gain=0.1, noise_stdev=0.1, distance=10,
                 wiring="crossed"):
        super().__init__()
        # Every Vehicle starts at the origin facing along +x -- the same
        # position and heading every repetition, every run. The Light is
        # always placed along +x too (see Light below), so this is also
        # always pointed directly at the light. The only thing that varies
        # between repetitions is the random noise. Keep that in mind if
        # you're ever tempted to run with noise_stdev=0.
        self.orientation = 0.0          # heading angle (radians); starts pointing along +x
        self.velocity = 0.0             # forward speed (arbitrary units)
        self.radius = 1.0               # body radius; also the sensor arm length

        self.left_sensor  = 0.0        # current left sensor activation  [0, 1]
        self.right_sensor = 0.0        # current right sensor activation [0, 1]
        self.left_motor   = 0.0        # left motor command
        self.right_motor  = 0.0        # right motor command

        # Sensor gain: normalise raw distance so that a sensor at `distance`
        # from the light reads approximately 0.5 (mid-range).
        # This scales raw Euclidean distance into a convenient sensor range.
        # Calibrated once, here, against the single light this Vehicle is
        # constructed with -- if you extend the simulator with a second
        # stimulus at a different distance (extra credit territory), it
        # will be read on a scale calibrated for the first one.
        self.sensor_gain = 1 / distance

        self.turn_gain   = turn_gain
        self.vel_gain    = 1 / 50       # converts average motor output to forward speed
        self.noise_stdev = noise_stdev
        self.angle_offset = angle_offset
        self.wiring = wiring

        # Sensor positions as Point objects, updated every step
        self.rs = Point()   # right sensor
        self.ls = Point()   # left sensor
        self.update_sensor_pos()

    def update_sensor_pos(self):
        """Recompute the world-frame positions of both sensors from the body pose."""
        self.rs.x_pos = self.x_pos + self.radius * np.cos(self.orientation + self.angle_offset)
        self.rs.y_pos = self.y_pos + self.radius * np.sin(self.orientation + self.angle_offset)
        self.ls.x_pos = self.x_pos + self.radius * np.cos(self.orientation - self.angle_offset)
        self.ls.y_pos = self.y_pos + self.radius * np.sin(self.orientation - self.angle_offset)

    def update_body_pos(self):
        """Advance the body one step along the current heading at the current velocity."""
        self.x_pos += self.velocity * np.cos(self.orientation)
        self.y_pos += self.velocity * np.sin(self.orientation)
        self.update_sensor_pos()

    def sense(self, light):
        """Read light intensity at each sensor position.

        Intensity follows an inverse-distance law: it is 1.0 when the sensor
        is at the light source and approaches 0 as distance grows.

        Args:
            light: A Light (or any Point) object representing the light source.
        """
        distance_right = self.sensor_gain * euclidean_distance(self.rs, light)
        distance_left  = self.sensor_gain * euclidean_distance(self.ls, light)
        self.right_sensor = 1.0 / (1.0 + distance_right)
        self.left_sensor  = 1.0 / (1.0 + distance_left)

    def think(self):
        """Map sensor activations to motor commands.

        TODO (Part 1 of the assignment): implement the sensor-to-motor
        wiring. Set self.left_motor and self.right_motor from
        self.left_sensor and self.right_sensor.

        Try a "crossed" (contralateral) wiring first: the left sensor
        drives the right motor, and the right sensor drives the left
        motor. Get the vehicle moving towards the light with this scheme
        before doing anything else.

        Once that works, try a "direct" (ipsilateral) wiring instead:
        each sensor drives the motor on the SAME side of the body (left
        sensor -> left motor, right sensor -> right motor). You can just
        edit the lines below by hand and re-run the simulator to compare
        the two behaviors.

        OPTIONAL (advanced): instead of hand-editing this method every
        time you want to switch schemes, use self.wiring (set in
        __init__, default "crossed") to select between the two schemes
        at runtime. A --wiring command-line flag is already wired up in
        both sim.py and study.py (it just constructs Vehicle(...,
        wiring=...)) — once think() reads self.wiring here, you can
        switch schemes with a flag instead of touching the code.
        """
        raise NotImplementedError(
            "Vehicle.think() is not implemented yet. See the TODO above "
            "(Part 1 of the assignment)."
        )

    def move(self):
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
        self.update_body_pos()

    def distance(self, light):
        """Return the Euclidean distance from the vehicle body centre to the light."""
        return euclidean_distance(self, light)


class Light(Point):
    """A stationary point light source.

    Args:
        distance: Distance from the origin at which the light is placed.
                  The light is positioned along the positive x-axis.
    """

    def __init__(self, distance=10.0):
        angle = 0.0   # light is placed along the +x axis
        super().__init__(x_pos=distance * np.cos(angle), y_pos=distance * np.sin(angle))