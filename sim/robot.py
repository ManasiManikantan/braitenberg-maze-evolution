"""
Braitenberg vehicle simulation.

Defines the core building blocks for a maze-navigation robot:
  - Point      : base class for any 2-D object with a position.
    - Vehicle    : a two-wheeled robot with two distance sensors and a differential drive.
    - Light      : a legacy stationary point light source.
"""

import numpy as np

# Modest robot-realism noise used by default when evolving controllers.
# Vehicle itself defaults every source to off so rule-based tests and other
# callers keep their exact behavior.
DEFAULT_ROBOT_NOISE = {
    "sensor_noise_stdev": 0.2,      # 2% of the default 10-unit sensor range
    "motor_noise_stdev": 0.05,      # per wheel, per step, in motor-command units
    "motor_mismatch": 0.05,         # left gain 1.025, right gain 0.975
    "start_position_stdev": 0.2,    # maze units, clipped to +/-0.4
    "start_heading_stdev": 0.1,     # radians (~6 degrees), clipped to +/-0.2
    "light_noise_stdev": 0.02,      # light intensity units (0-1 scale), like 2% sensor noise
}


DEFAULT_SIDE_SENSOR_ANGLE = np.deg2rad(60.0)
DEFAULT_LIGHT_SENSOR_ANGLE = np.deg2rad(45.0)
DEFAULT_LIGHT_FALLOFF_DISTANCE = 5.0
LIGHT_OCCLUSION_MODES = ("blocked", "through")


def euclidean_distance(point1, point2):
    """Return the Euclidean distance between two Point objects."""
    return np.sqrt((point1.x_pos - point2.x_pos)**2 + (point1.y_pos - point2.y_pos)**2)


class Point:
    """A 2-D object with an (x, y) position."""

    def __init__(self, x_pos=0.0, y_pos=0.0):
        self.x_pos = x_pos
        self.y_pos = y_pos


class Vehicle(Point):
    """A differential-drive vehicle with three distance sensors.

    The vehicle body is modelled as a circle. The front sensor sits at the
    front of the body and looks straight ahead. The left and right sensors
    sit on the body's left and right edges and look outward at
    ``side_sensor_angle`` from the heading (0 makes all three rays parallel,
    the original layout).

    Args:
        angle_offset: Legacy argument retained for constructor compatibility
                      (unused; see ``side_sensor_angle``).
        side_sensor_angle: Angle (radians) between the heading and each side
                      ray, from 0 (parallel) to pi/2. Default 60 degrees.
        start_position: Starting (x, y) of the body centre. Default origin.
        start_heading: Starting heading in radians. Default 0 (+x).
        light_occlusion: None (default, no light sensors), "blocked" (walls
                      block the light from the goal), or "through" (walls do
                      not). When set, the controller receives five inputs:
                      left, front, right distance, then left and right light.
        light_sensor_angle: Angle (radians) of each light sensor from the
                      heading; sensors sit on the body edge at +/- this angle
                      and face outward along it. Default 45 degrees.
        light_falloff_distance: Distance at which intensity halves before
                      the angular factor: I = max(0, cos a) / (1 + (d / d0)^2).
        light_noise_stdev: Gaussian noise added to each light reading (then
                      clipped to [0, 1]). Zero disables it.
        turn_gain:    How strongly the sensor difference steers the vehicle.
                      Larger values produce sharper turns.
        noise_stdev:  Standard deviation of Gaussian noise added to the
                      orientation at each step.  Zero means no noise.
        sensor_range: Maximum sensor distance in maze units.
        avoidance_distance: Distance at which the reactive safety layer steers away.
        wall_avoidance_distance: Clearance margin for fixed maze walls.
        reverse_steps: Number of steps in a collision-recovery reverse burst.
        collision_reverse_steps: Short reverse burst after first wall contact.
        reverse_speed: Absolute motor command used during reverse recovery.
        cruise_speed: Minimum forward motor command when the path is clear.
        turn_speed: Motor command magnitude used to turn away from blocked rays.
        goal_seek_distance: Maximum distance for direct goal-seeking behavior.
        wiring:       Sensor-to-motor mapping: "crossed" (default) or "direct".
        controller:   Optional controller with a two-input, two-output API.
        distance:     Legacy alias for sensor_range.
        sensor_noise_stdev: Gaussian noise (maze units) added to each ray
                      distance reading every step. Zero disables it.
        motor_noise_stdev: Gaussian noise added independently to each wheel's
                      motor command every step. Zero disables it.
        motor_mismatch: Fixed fractional gain difference between the wheels:
                      left gain is 1 + m/2 and right gain is 1 - m/2.
        start_position_stdev: Gaussian jitter (maze units) of the start x and y,
                      clipped to two standard deviations.
        start_heading_stdev: Gaussian jitter (radians) of the start heading,
                      clipped to two standard deviations.

    All noise sources default to off; ``DEFAULT_ROBOT_NOISE`` holds the
    modest levels used by default for evolution. Disabled sources draw no
    random numbers, so seeded runs without them are unchanged.
    """

    def __init__(self, angle_offset=np.pi/6, turn_gain=0.1, noise_stdev=0.1,
                 sensor_range=10, wiring="crossed", controller=None, distance=None,
                 avoidance_distance=3.0, reverse_steps=12, collision_reverse_steps=3,
                 reverse_speed=0.75,
                 avoidance_gain=1.5, cruise_speed=0.5, turn_speed=0.75,
                 goal_seek_distance=4.0, wall_avoidance_distance=0.25,
                 navigation_heuristics=True, noise_seed=None,
                 sensor_noise_stdev=0.0, motor_noise_stdev=0.0, motor_mismatch=0.0,
                 start_position_stdev=0.0, start_heading_stdev=0.0,
                 side_sensor_angle=DEFAULT_SIDE_SENSOR_ANGLE,
                 start_position=(0.0, 0.0), start_heading=0.0,
                 light_occlusion=None, light_sensor_angle=DEFAULT_LIGHT_SENSOR_ANGLE,
                 light_falloff_distance=DEFAULT_LIGHT_FALLOFF_DISTANCE, light_noise_stdev=0.0):
        super().__init__(float(start_position[0]), float(start_position[1]))
        # The vehicle starts at start_position facing start_heading (default:
        # the origin, facing +x), optionally jittered by start_position_stdev /
        # start_heading_stdev below.
        self.orientation = float(start_heading)  # heading angle (radians)
        self.velocity = 0.0             # forward speed (arbitrary units)
        self.radius = 1.0               # body radius; also the sensor arm length

        self.left_sensor  = 0.0        # current left sensor activation  [0, 1]
        self.front_sensor = 0.0        # current forward sensor activation [0, 1]
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
        if wall_avoidance_distance <= 0:
            raise ValueError("wall_avoidance_distance must be positive")
        if reverse_steps < 1:
            raise ValueError("reverse_steps must be at least 1")
        if collision_reverse_steps < 1:
            raise ValueError("collision_reverse_steps must be at least 1")
        if not 0 < reverse_speed <= 1:
            raise ValueError("reverse_speed must be in (0, 1]")
        if avoidance_gain < 0:
            raise ValueError("avoidance_gain must be non-negative")
        if not 0 < cruise_speed <= 1:
            raise ValueError("cruise_speed must be in (0, 1]")
        if not 0 < turn_speed <= 1:
            raise ValueError("turn_speed must be in (0, 1]")
        if goal_seek_distance <= 0:
            raise ValueError("goal_seek_distance must be positive")
        if min(sensor_noise_stdev, motor_noise_stdev,
               start_position_stdev, start_heading_stdev) < 0:
            raise ValueError("noise standard deviations must be non-negative")
        if not -1 < motor_mismatch < 1:
            raise ValueError("motor_mismatch must be in (-1, 1)")
        if not 0 <= side_sensor_angle <= np.pi / 2:
            raise ValueError("side_sensor_angle must be between 0 and pi/2 radians")
        self.side_sensor_angle = float(side_sensor_angle)
        if light_occlusion is not None and light_occlusion not in LIGHT_OCCLUSION_MODES:
            raise ValueError("light_occlusion must be None, 'blocked', or 'through'")
        if not 0 <= light_sensor_angle <= np.pi / 2:
            raise ValueError("light_sensor_angle must be between 0 and pi/2 radians")
        if light_falloff_distance <= 0 or light_noise_stdev < 0:
            raise ValueError("light_falloff_distance must be positive and light noise non-negative")
        self.light_occlusion = light_occlusion
        self.light_sensor_angle = float(light_sensor_angle)
        self.light_falloff_distance = float(light_falloff_distance)
        self.light_noise_stdev = light_noise_stdev
        self.left_light = 0.0
        self.right_light = 0.0
        self.sensor_noise_stdev = sensor_noise_stdev
        self.motor_noise_stdev = motor_noise_stdev
        self.motor_mismatch = motor_mismatch
        self.sensor_range = sensor_range
        self.avoidance_distance = avoidance_distance
        self.wall_avoidance_distance = wall_avoidance_distance
        self.reverse_steps = reverse_steps
        self.collision_reverse_steps = collision_reverse_steps
        self.reverse_speed = reverse_speed
        self.avoidance_gain = avoidance_gain
        self.cruise_speed = cruise_speed
        self.turn_speed = turn_speed
        self.goal_seek_distance = goal_seek_distance
        self.navigation_heuristics = navigation_heuristics
        self._noise_rng = np.random.default_rng(noise_seed)
        if start_position_stdev > 0:
            offset_x, offset_y = map(float, np.clip(
                self._noise_rng.normal(0.0, start_position_stdev, 2),
                -2 * start_position_stdev,
                2 * start_position_stdev,
            ))
            self.x_pos += offset_x
            self.y_pos += offset_y
        if start_heading_stdev > 0:
            self.orientation += float(np.clip(
                self._noise_rng.normal(0.0, start_heading_stdev),
                -2 * start_heading_stdev,
                2 * start_heading_stdev,
            ))
        self.left_distance = sensor_range
        self.front_distance = sensor_range
        self.right_distance = sensor_range
        self.left_hit_type = None
        self.front_hit_type = None
        self.right_hit_type = None
        self._reverse_steps_remaining = 0
        self._turning_to_clear = False
        self._scan_rotation = 0.0
        self._scan_center_heading = 0.0
        self._scan_phase = "sweep"
        self._scan_best_distance = -np.inf
        self._scan_best_heading = 0.0
        self._aligning_to_opening = False
        self._opening_heading = 0.0
        self.goal_visible = False
        self._goal_distance = np.inf
        self._goal_bearing = 0.0
        self._turn_direction = 1

        self.turn_gain   = turn_gain
        self.vel_gain    = 1 / 15       # converts average motor output to forward speed
        self.noise_stdev = noise_stdev
        self.angle_offset = angle_offset
        self.wiring = wiring
        self.controller = controller

        # Sensor positions as Point objects, updated every step
        self.rs = Point()   # right sensor
        self.fs = Point()   # forward sensor
        self.ls = Point()   # left sensor
        self.update_sensor_pos()

    def update_sensor_pos(self):
        """Place side sensors on the body edge and all sensors facing forward."""
        forward_x = np.cos(self.orientation)
        forward_y = np.sin(self.orientation)
        left_x = -np.sin(self.orientation)
        left_y = np.cos(self.orientation)
        self.ls.x_pos = self.x_pos + self.radius * left_x
        self.ls.y_pos = self.y_pos + self.radius * left_y
        self.fs.x_pos = self.x_pos + self.radius * forward_x
        self.fs.y_pos = self.y_pos + self.radius * forward_y
        self.rs.x_pos = self.x_pos - self.radius * left_x
        self.rs.y_pos = self.y_pos - self.radius * left_y

    def update_body_pos(self):
        """Advance the body one step along the current heading at the current velocity."""
        self.x_pos += self.velocity * np.cos(self.orientation)
        self.y_pos += self.velocity * np.sin(self.orientation)
        self.update_sensor_pos()

    def sensor_angles(self):
        """Return world-frame ray angles for the left, front, and right sensors."""
        return (
            self.orientation + self.side_sensor_angle,
            self.orientation,
            self.orientation - self.side_sensor_angle,
        )

    def controller_inputs(self):
        """Inputs passed to the controller: 3 distance readings, plus 2 light readings if enabled."""
        inputs = (self.left_sensor, self.front_sensor, self.right_sensor)
        if self.light_occlusion is not None:
            inputs += (self.left_light, self.right_light)
        return inputs

    def light_sensor_poses(self):
        """Return ((x, y, facing_angle) for the left light sensor, same for the right)."""
        poses = []
        for side in (1, -1):
            angle = self.orientation + side * self.light_sensor_angle
            poses.append((self.x_pos + self.radius * np.cos(angle),
                          self.y_pos + self.radius * np.sin(angle), angle))
        return poses

    def _sense_light(self, maze, goal):
        goal_x, goal_y = (goal.x_pos, goal.y_pos) if hasattr(goal, "x_pos") else goal
        readings = []
        for sensor_x, sensor_y, facing in self.light_sensor_poses():
            dx, dy = goal_x - sensor_x, goal_y - sensor_y
            distance = float(np.hypot(dx, dy))
            if distance < 1e-9:
                intensity = 1.0
            else:
                cosine = (dx * np.cos(facing) + dy * np.sin(facing)) / distance
                intensity = max(0.0, cosine) / (1.0 + (distance / self.light_falloff_distance)**2)
                if (self.light_occlusion == "blocked" and intensity > 0.0
                        and maze.raycast((sensor_x, sensor_y), (dx, dy), distance) < distance - 1e-6):
                    intensity = 0.0
            readings.append(intensity)
        if self.light_noise_stdev > 0:
            readings = np.array(readings) + self._noise_rng.normal(0.0, self.light_noise_stdev, 2)
        return tuple(float(np.clip(value, 0.0, 1.0)) for value in readings)

    def sense(self, maze, goal=None):
        """Read obstacle proximity and check whether the goal is visible.

        Sensor values range from 0 (nothing detected within range) to 1
        (an obstacle at the sensor origin). Raw distances are also exposed as
        ``left_distance`` and ``right_distance``.

        Args:
            maze: Maze providing a ``raycast`` method.
        """
        left_angle, front_angle, right_angle = self.sensor_angles()
        self.left_distance, self.left_hit_type = maze.raycast_with_type(
            self.ls,
            (np.cos(left_angle), np.sin(left_angle)),
            self.sensor_range,
        )
        self.front_distance, self.front_hit_type = maze.raycast_with_type(
            self.fs,
            (np.cos(front_angle), np.sin(front_angle)),
            self.sensor_range,
        )
        self.right_distance, self.right_hit_type = maze.raycast_with_type(
            self.rs,
            (np.cos(right_angle), np.sin(right_angle)),
            self.sensor_range,
        )
        if self.sensor_noise_stdev > 0:
            noisy = np.clip(
                np.array([self.left_distance, self.front_distance, self.right_distance])
                + self._noise_rng.normal(0.0, self.sensor_noise_stdev, 3),
                0.0,
                self.sensor_range,
            )
            self.left_distance, self.front_distance, self.right_distance = map(float, noisy)
        self.left_sensor = 1.0 - self.left_distance / self.sensor_range
        self.front_sensor = 1.0 - self.front_distance / self.sensor_range
        self.right_sensor = 1.0 - self.right_distance / self.sensor_range
        if self.light_occlusion is not None and goal is not None:
            self.left_light, self.right_light = self._sense_light(maze, goal)
        self.goal_visible = False
        self._goal_distance = np.inf
        if goal is not None:
            if hasattr(goal, "x_pos"):
                goal_x, goal_y = goal.x_pos, goal.y_pos
            else:
                goal_x, goal_y = goal
            goal_dx = goal_x - self.x_pos
            goal_dy = goal_y - self.y_pos
            goal_distance = np.hypot(goal_dx, goal_dy)
            self._goal_distance = goal_distance
            if goal_distance > 0:
                self._goal_bearing = np.arctan2(goal_dy, goal_dx)
                distance_to_goal = maze.raycast(
                    self,
                    (goal_dx, goal_dy),
                    goal_distance,
                )
                self.goal_visible = distance_to_goal >= goal_distance - 1e-6

    def _required_clearance(self, hit_type):
        if hit_type == "wall":
            return self.wall_avoidance_distance
        return self.avoidance_distance

    def _hazard(self, clearance, hit_type):
        required_clearance = self._required_clearance(hit_type)
        return float(np.clip(1.0 - clearance / required_clearance, 0.0, 1.0))

    def think(self):
        """Map sensors to motors, then apply a proximity-based safety override."""
        left_clearance = max(0.0, self.left_distance - self.radius)
        front_clearance = self.front_distance
        right_clearance = max(0.0, self.right_distance - self.radius)
        if self.navigation_heuristics and self._reverse_steps_remaining > 0:
            self.left_motor = self.right_motor = -self.reverse_speed
            return

        if (
            self.navigation_heuristics
            and self.goal_visible
            and self._goal_distance <= self.goal_seek_distance
        ):
            self._turning_to_clear = False
            self._aligning_to_opening = False
            heading_error = np.arctan2(
                np.sin(self._goal_bearing - self.orientation),
                np.cos(self._goal_bearing - self.orientation),
            )
            turn = self.turn_speed * float(np.clip(
                heading_error / (np.pi / 2), -1.0, 1.0
            ))
            left_hazard = self._hazard(left_clearance, self.left_hit_type)
            front_hazard = self._hazard(front_clearance, self.front_hit_type)
            right_hazard = self._hazard(right_clearance, self.right_hit_type)
            self.left_motor = float(np.clip(
                self.cruise_speed + turn - self.avoidance_gain * max(left_hazard, front_hazard),
                -1.0,
                1.0,
            ))
            self.right_motor = float(np.clip(
                self.cruise_speed - turn - self.avoidance_gain * max(right_hazard, front_hazard),
                -1.0,
                1.0,
            ))
            return

        if self.navigation_heuristics and self._aligning_to_opening:
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

        if self.navigation_heuristics and self._turning_to_clear:
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
                if self._scan_best_distance <= 0.0:
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
            left_clearance <= self._required_clearance(self.left_hit_type)
            and right_clearance <= self._required_clearance(self.right_hit_type)
        )
        front_blocked = front_clearance <= self._required_clearance(self.front_hit_type)
        if (
            self.navigation_heuristics
            and (both_rays_blocked or front_blocked)
            and not self._turning_to_clear
        ):
            self._begin_avoidance_turn()
            self._record_scan_opening()
            self.left_motor = self.turn_speed * self._turn_direction
            self.right_motor = -self.turn_speed * self._turn_direction
            return

        if self.controller is not None:
            left_motor, right_motor = self.controller.forward(self.controller_inputs())
        elif self.wiring == "crossed":
            left_motor = self.right_sensor
            right_motor = self.left_sensor
        elif self.wiring == "direct":
            left_motor = self.left_sensor
            right_motor = self.right_sensor
        else:
            raise ValueError("wiring must be 'crossed' or 'direct'")

        if not self.navigation_heuristics:
            self.left_motor = float(np.clip(left_motor, -1.0, 1.0))
            self.right_motor = float(np.clip(right_motor, -1.0, 1.0))
            return

        left_hazard = self._hazard(left_clearance, self.left_hit_type)
        right_hazard = self._hazard(right_clearance, self.right_hit_type)
        front_hazard = self._hazard(front_clearance, self.front_hit_type)
        opening_difference = left_clearance - right_clearance
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

        if max(left_hazard, front_hazard, right_hazard) == 0.0:
            left_motor = max(float(left_motor), self.cruise_speed)
            right_motor = max(float(right_motor), self.cruise_speed)
        self.left_motor = float(np.clip(
            left_motor - self.avoidance_gain * max(left_hazard, front_hazard), -1.0, 1.0
        ))
        self.right_motor = float(np.clip(
            right_motor - self.avoidance_gain * max(right_hazard, front_hazard), -1.0, 1.0
        ))


    def _begin_avoidance_turn(self):
        self._turning_to_clear = True
        self._scan_rotation = 0.0
        self._scan_center_heading = self.orientation
        self._scan_phase = "position"
        self._scan_best_distance = -np.inf
        self._scan_best_heading = self.orientation
        self._turn_direction = -1
        self._aligning_to_opening = False

    def request_avoidance_scan(self):
        """Cancel recovery motion and scan ahead after entering a known path cell."""
        if not self.navigation_heuristics:
            return
        self._reverse_steps_remaining = 0
        self._begin_avoidance_turn()

    def _record_scan_opening(self):
        if self._scan_phase != "sweep":
            return
        left_margin = max(0.0, self.left_distance - self.radius) - self._required_clearance(
            self.left_hit_type
        )
        front_margin = self.front_distance - self._required_clearance(self.front_hit_type)
        right_margin = max(0.0, self.right_distance - self.radius) - self._required_clearance(
            self.right_hit_type
        )
        opening_margin = min(left_margin, front_margin, right_margin)
        if opening_margin > self._scan_best_distance:
            self._scan_best_distance = opening_margin
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
        noise = self._noise_rng.normal(0, self.noise_stdev)
        left_wheel = self.left_motor * (1 + self.motor_mismatch / 2)
        right_wheel = self.right_motor * (1 - self.motor_mismatch / 2)
        if self.motor_noise_stdev > 0:
            left_noise, right_noise = self._noise_rng.normal(0.0, self.motor_noise_stdev, 2)
            left_wheel += left_noise
            right_wheel += right_noise
        self.orientation += self.turn_gain * (left_wheel - right_wheel) + noise
        self.velocity = self.vel_gain * ((right_wheel + left_wheel) / 2)
        old_position = (self.x_pos, self.y_pos)
        self.update_body_pos()
        collided = maze is not None and maze.collides(
            (self.x_pos, self.y_pos), self.radius
        )
        if collided:
            self.x_pos, self.y_pos = old_position
            self.velocity = 0.0
            if was_reversing:
                self._reverse_steps_remaining = 0
                self._begin_avoidance_turn()
            elif self.navigation_heuristics:
                self._reverse_steps_remaining = 0
                self._turning_to_clear = False
                self._aligning_to_opening = False
                self.reverse(self.collision_reverse_steps)
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
