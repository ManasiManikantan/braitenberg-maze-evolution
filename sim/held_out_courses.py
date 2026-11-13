"""Hand-designed 12 x 12 loop-free test courses (held out; never used for training).

Each course is a 3 x 3 grid of 4-unit cells (corridors 4 units wide, twice the
robot's diameter) carved as a perfect maze, so every wall is connected to the
boundary. Cells are (column, row) with column 0 on the left (x from -6 to -2)
and row 0 at the bottom (y from -6 to -2). The robot starts at the centre of
``start_cell`` facing ``start_heading``; the goal is the centre of
``goal_cell``.

- ``A``: the route runs up the left column, along the top and down the right
  column. A two-cell dead end hangs off the route's right side, so a
  left-wall follower reaches the goal without detours and a right-wall
  follower has to explore the dead end ("left-favoring").
- ``B``: the left-right mirror of A ("right-favoring").
- ``C``: a straight route up the centre column with mirror-image side
  branches on both sides, so left- and right-wall followers each explore one
  branch ("neutral").
"""

import math

from .maze_generator import maze_from_passages

CELL_SIZE = 4.0
COLUMNS = ROWS = 3

_A_PASSAGES = (
    ((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (1, 2)), ((1, 2), (2, 2)),
    ((2, 2), (2, 1)), ((2, 1), (2, 0)),
    ((1, 2), (1, 1)), ((1, 1), (1, 0)),   # dead end on the route's right side
)


def _mirror(cell):
    return (COLUMNS - 1 - cell[0], cell[1])


COURSE_SPECS = {
    "A": {
        "passages": _A_PASSAGES,
        "start_cell": (0, 0),
        "start_heading": math.pi / 2,
        "goal_cell": (2, 0),
    },
    "B": {
        "passages": tuple((_mirror(a), _mirror(b)) for a, b in _A_PASSAGES),
        "start_cell": (2, 0),
        "start_heading": math.pi / 2,
        "goal_cell": (0, 0),
    },
    "C": {
        "passages": (
            ((1, 0), (1, 1)), ((1, 1), (1, 2)),
            ((1, 1), (0, 1)), ((0, 1), (0, 0)), ((0, 1), (0, 2)),
            ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((2, 1), (2, 2)),
        ),
        "start_cell": (1, 0),
        "start_heading": math.pi / 2,
        "goal_cell": (1, 2),
    },
}
TEST_COURSE_NAMES = tuple(COURSE_SPECS)


def make_test_course(name):
    """Return ``(maze, start_position, start_heading)`` for test course ``name``."""
    spec = COURSE_SPECS[name]
    maze, start = maze_from_passages(
        spec["passages"], COLUMNS, ROWS, CELL_SIZE, spec["goal_cell"], spec["start_cell"]
    )
    return maze, start, spec["start_heading"]


__all__ = ["COURSE_SPECS", "TEST_COURSE_NAMES", "make_test_course"]
