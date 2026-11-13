"""Hand-written left/right wall followers using only the 3 distance sensors.

Same controller interface as the evolved networks: forward(inputs) receives
(left, front, right) proximities p = 1 - distance / sensor_range and returns
(left_motor, right_motor). Physical convention: heading += k * (R - L), so a
faster right wheel turns the robot LEFT.

Geometry: a side ray starts on the body edge and points 60 degrees off the
heading, so a parallel wall whose distance from the body centre is D reads
(D - 1) / sin(60 deg); the middle of a 4-unit corridor (D = 2) reads 1.15.

Rules (described for the left follower; the right one is the mirror image):
  1. front ray < front_stop          -> pivot right (away from the wall)
  2. left ray  < touching            -> turn right hard (off by default: touching=0
                                        made validation results worse)
     left ray  < near                -> veer right gently
  3. left ray  > far (lost / opening) -> arc left with radius ~2 units, which
     keeps the body ~1.4 units clear of a wall tip while going around it
  4. otherwise                       -> drive straight
"""

import numpy as np

SENSOR_RANGE = 10.0


def arc_commands(radius, outer=0.6):
    """Motor pair (outer, inner) giving a turning radius in maze units.

    Speed = (outer + inner) / 2 / 15 per step and turn rate = 0.1 (outer - inner)
    rad per step, so radius = (outer + inner) / (3 (outer - inner)).
    """
    inner = outer * (3 * radius - 1) / (3 * radius + 1)
    return outer, inner


class WallFollower:
    def __init__(self, side="left", front_stop=1.6, near=0.8, far=2.0, arc_radius=2.0,
                 cruise=0.6, pivot=0.45, veer=0.15, touching=0.0):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.touching = touching
        self.front_stop = front_stop
        self.near = near
        self.far = far
        self.arc = arc_commands(arc_radius, cruise)
        self.cruise = cruise
        self.pivot = pivot
        self.veer = veer

    def forward(self, inputs):
        left_p, front_p, right_p = inputs[:3]
        left, front, right = (SENSOR_RANGE * (1.0 - p) for p in (left_p, front_p, right_p))
        followed = left if self.side == "left" else right
        # (motor on the followed side, motor on the other side)
        if front < self.front_stop:
            followed_motor, other_motor = -self.pivot, self.pivot      # pivot away
        elif followed < self.touching:
            followed_motor, other_motor = 0.0, self.pivot              # hard turn away
        elif followed < self.near:
            followed_motor, other_motor = self.cruise - self.veer, self.cruise  # veer away
        elif followed > self.far:
            outer, inner = self.arc
            followed_motor, other_motor = outer, inner                # arc toward
        else:
            followed_motor, other_motor = self.cruise, self.cruise
        # Physical convention (as on a real robot): a faster RIGHT wheel turns
        # the robot LEFT. "followed_motor" is the command that steers toward
        # the followed wall when it is larger, so it drives the OPPOSITE wheel.
        if self.side == "left":
            return np.array([other_motor, followed_motor])
        return np.array([followed_motor, other_motor])
