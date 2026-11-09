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
        avoidance_distance: Distance at which the reactive safety layer steers away.
        reverse_steps: Number of steps in a collision-recovery reverse burst.
        reverse_speed: Absolute motor command used during reverse recovery.
        cruise_speed: Minimum forward motor command when the path is clear.
        turn_speed: Motor command magnitude used to turn away from blocked rays.
        wiring:       Sensor-to-motor mapping: "crossed" (default) or "direct".
        controller:   Optional controller with a two-input, two-output API.
        distance:     Legacy alias for sensor_range.
    """

    def __init__(self, angle_offset=np.pi/6, turn_gain=0.1, noise_stdev=0.1,
                 sensor_range=10, wiring="crossed", controller=None, distance=None,
                 avoidance_distance=3.0, reverse_steps=12, reverse_speed=0.75,
                 avoidance_gain=1.5, cruise_speed=0.5, turn_speed=0.75):
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
        if avoidance_distance <= 0:
            raise ValueError("avoidance_distance must be positive")
        if avoidance_distance > sensor_range:
            raise ValueError("avoidance_distance cannot exceed sensor_range")
        if reverse_steps < 1:
            raise ValueError("reverse_steps must be at least 1")
        if not 0 < reverse_speed <= 1:
            raise ValueError("reverse_speed must be in (0, 1]")
        if avoidance_gain < 0:
            raise ValueError("avoidance_gain must be non-negative")
        if not 0 < cruise_speed <= 1:
            raise ValueError("cruise_speed must be in (0, 1]")
        if not 0 < turn_speed <= 1:
            raise ValueError("turn_speed must be in (0, 1]")
        self.sensor_range = sensor_range
        self.avoidance_distance = avoidance_distance
        self.reverse_steps = reverse_steps
        self.reverse_speed = reverse_speed
        self.avoidance_gain = avoidance_gain
        self.cruise_speed = cruise_speed
        self.turn_speed = turn_speed
        self.left_distance = sensor_range
        self.right_distance = sensor_range
        self._reverse_steps_remaining = 0
        self._turning_to_clear = False
        self._scan_rotation = 0.0
        self._scan_center_heading = 0.0
        self._scan_phase = "sweep"
        self._scan_best_distance = -1.0
        self._scan_best_heading = 0.0
        self._aligning_to_opening = False
        self._opening_heading = 0.0
        self._turn_direction = 1

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
        """Map sensors to motors, then apply a proximity-based safety override."""
        if self._reverse_steps_remaining > 0:
            self.left_motor = self.right_motor = -self.reverse_speed
            return

        if self._aligning_to_opening:
            heading_error = np.arctan2(
                np.sin(self._opening_heading - self.orientation),
                np.cos(self._opening_heading - self.orientation),
            )
            if abs(heading_error) > 0.08:
                turn_direction = 1 if heading_error > 0 else -1
                self.left_motor = self.turn_speed * turn_direction
                self.right_motor = -self.turn_speed * turn_direction
                return
            self._aligning_to_opening = False

        if self._turning_to_clear:
            if self._scan_phase == "position":
                scan_start = self._scan_center_heading - np.pi / 2
                heading_error = np.arctan2(
                    np.sin(scan_start - self.orientation),
                    np.cos(scan_start - self.orientation),
                )
                if abs(heading_error) > 0.02:
                    turn_direction = 1 if heading_error > 0 else -1
                    turn_magnitude = self.turn_speed
                    if self.turn_gain > 0:
                        turn_magnitude = min(
                            turn_magnitude,
                            abs(heading_error) / (2 * self.turn_gain),
                        )
                    self.left_motor = turn_magnitude * turn_direction
                    self.right_motor = -turn_magnitude * turn_direction
                    return
                self._scan_phase = "sweep"
                self._scan_rotation = 0.0
                self._turn_direction = 1

            self._record_scan_opening()
            remaining_rotation = np.pi - self._scan_rotation
            if remaining_rotation <= 0.04:
                self._turning_to_clear = False
                if self._scan_best_distance <= self.avoidance_distance:
                    self.reverse()
                    self.left_motor = self.right_motor = -self.reverse_speed
                    return
                self._opening_heading = self._scan_best_heading
                self._aligning_to_opening = True
                return self.think()
            turn_magnitude = self.turn_speed
            if self.turn_gain > 0:
                turn_magnitude = min(turn_magnitude, remaining_rotation / (2 * self.turn_gain))
            self.left_motor = turn_magnitude * self._turn_direction
            self.right_motor = -turn_magnitude * self._turn_direction
            return

        both_rays_blocked = (
            self.left_distance <= self.avoidance_distance
            and self.right_distance <= self.avoidance_distance
        )
        if both_rays_blocked and not self._turning_to_clear:
            self._begin_avoidance_turn()
            self._record_scan_opening()
            self.left_motor = self.turn_speed * self._turn_direction
            self.right_motor = -self.turn_speed * self._turn_direction
            return

        if self.controller is not None:
            left_motor, right_motor = self.controller.forward(
                (self.left_sensor, self.right_sensor)
            )
        elif self.wiring == "crossed":
            left_motor = self.right_sensor
            right_motor = self.left_sensor
        elif self.wiring == "direct":
            left_motor = self.left_sensor
            right_motor = self.right_sensor
        else:
            raise ValueError("wiring must be 'crossed' or 'direct'")

        left_hazard = float(np.clip(
            1.0 - self.left_distance / self.avoidance_distance, 0.0, 1.0
        ))
        right_hazard = float(np.clip(
            1.0 - self.right_distance / self.avoidance_distance, 0.0, 1.0
        ))
        opening_difference = self.left_distance - self.right_distance
        if abs(opening_difference) >= 0.25:
            base_speed = max(
                self.cruise_speed,
                (float(left_motor) + float(right_motor)) / 2,
            )
            opening_turn = self.turn_speed * float(np.clip(
                opening_difference / self.avoidance_distance, -1.0, 1.0
            ))
            self.left_motor = float(np.clip(base_speed + opening_turn, -1.0, 1.0))
            self.right_motor = float(np.clip(base_speed - opening_turn, -1.0, 1.0))
            return

        if max(left_hazard, right_hazard) == 0.0:
            left_motor = max(float(left_motor), self.cruise_speed)
            right_motor = max(float(right_motor), self.cruise_speed)
        self.left_motor = float(np.clip(
            left_motor - self.avoidance_gain * left_hazard, -1.0, 1.0
        ))
        self.right_motor = float(np.clip(
            right_motor - self.avoidance_gain * right_hazard, -1.0, 1.0
        ))


    def _begin_avoidance_turn(self):
        self._turning_to_clear = True
        self._scan_rotation = 0.0
        self._scan_center_heading = self.orientation
        self._scan_phase = "position"
        self._scan_best_distance = -1.0
        self._scan_best_heading = self.orientation
        self._turn_direction = -1
        self._aligning_to_opening = False

    def _record_scan_opening(self):
        if self._scan_phase != "sweep":
            return
        best_distance = max(self.left_distance, self.right_distance)
        if best_distance > self._scan_best_distance:
            self._scan_best_distance = best_distance
            self._scan_best_heading = self.orientation

    def reverse(self, steps=None):
        """Schedule a straight reverse burst for the next movement steps."""
        reverse_steps = self.reverse_steps if steps is None else steps
        if reverse_steps < 1:
            raise ValueError("steps must be at least 1")
        self._reverse_steps_remaining = reverse_steps

    def move(self, maze=None):
        """Update orientation and velocity, then advance the body position.

        Differential drive kinematics:
          - Turning rate  = turn_gain × (left_motor − right_motor) + noise
          - Forward speed = vel_gain  × average(left_motor, right_motor)

        Adding noise to the orientation simulates imperfect actuators and
        prevents the vehicle from following a perfectly straight trajectory.
        """
        was_reversing = self._reverse_steps_remaining > 0
        was_scanning = self._turning_to_clear
        old_orientation = self.orientation
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
            self.velocity = 0.0
            self._reverse_steps_remaining = 0
            self._begin_avoidance_turn()
            self.update_sensor_pos()
        elif was_reversing:
            self._reverse_steps_remaining -= 1
        elif was_scanning and self._scan_phase == "sweep":
            angle_delta = (self.orientation - old_orientation + np.pi) % (2 * np.pi) - np.pi
            self._scan_rotation += abs(angle_delta)
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
