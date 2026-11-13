"""Left-right mirror image of a controller."""

import numpy as np


class MirroredController:
    """Wrap a controller so it behaves as its left-right mirror image.

    The left and right distance inputs (and, with five inputs, the left and
    right light inputs) are swapped before the wrapped controller sees them,
    and its left and right motor outputs are swapped afterwards. In the
    mirror image of a maze the mirrored controller therefore drives the mirror
    image of the original's path.
    """

    def __init__(self, controller):
        self.controller = controller

    def forward(self, inputs):
        inputs = np.asarray(inputs, dtype=float)
        mirrored = inputs.copy()
        mirrored[0], mirrored[2] = inputs[2], inputs[0]
        if len(inputs) >= 5:
            mirrored[3], mirrored[4] = inputs[4], inputs[3]
        left_motor, right_motor = self.controller.forward(mirrored)
        return np.array([right_motor, left_motor])

    __call__ = forward


__all__ = ["MirroredController"]
