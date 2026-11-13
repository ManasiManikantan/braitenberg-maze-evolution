"""Narrowest gaps between separate solids (walls, obstacles, boundary) in each layout."""
import sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import itertools, math
import numpy as np
from sim.maze import Maze
from sim.maze_generator import generate_perfect_maze
from evolution.experiment import DEFAULT_CURRICULUM


def seg_dist(a, b):
    def pt(p, s):
        return math.sqrt(Maze._point_segment_distance_squared(p, s))
    (p1, p2), (q1, q2) = a, b
    # segments intersect? (proper or touching) -> 0
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return 0.0
    return min(pt(p1, b), pt(p2, b), pt(q1, a), pt(q2, a))


def solids(maze):
    """Group segments into connected solids (touching segments merge)."""
    pieces = [[w] for w in maze.walls] + [list(Maze._obstacle_walls(o)) for o in maze.obstacles]
    merged = True
    while merged:
        merged = False
        for i, j in itertools.combinations(range(len(pieces)), 2):
            if min(seg_dist(a, b) for a in pieces[i] for b in pieces[j]) < 1e-9:
                pieces[i] += pieces.pop(j)
                merged = True
                break
    return pieces


def gaps(maze):
    pieces = solids(maze)
    out = []
    for i, j in itertools.combinations(range(len(pieces)), 2):
        out.append(min(seg_dist(a, b) for a in pieces[i] for b in pieces[j]))
    return sorted(out), len(pieces)


for d in (1, 2, 3, 4):
    g, n = gaps(Maze(difficulty=d))
    print(f"course_{d}: {n} separate solids; narrowest gaps {[round(x, 2) for x in g[:4]]}; "
          f"gaps < 2.0 (robot can't pass): {sum(x < 2.0 for x in g)}, 2.0-3.0 (tight): {sum(2.0 <= x < 3.0 for x in g)}")
for _, opts in DEFAULT_CURRICULUM:
    rng = np.random.default_rng(2026)
    stats = [gaps(generate_perfect_maze(rng, **opts).maze) for _ in range(20)]
    print(f"random {opts['size']:.0f}x{opts['size']:.0f}: separate solids per maze {sorted({n for _, n in stats})}; "
          f"narrowest gap over 20 mazes {min(min(g) for g, _ in stats if g) if any(g for g, _ in stats) else 'none (all walls connected)'}")
