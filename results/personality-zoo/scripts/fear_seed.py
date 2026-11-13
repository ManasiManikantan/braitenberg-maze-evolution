"""Hand-set 'fearful' controller (5 inputs, hidden 4) and a check against GA winners.

Hidden units: three steep "danger" detectors (one per distance sensor, ~0 below
THRESHOLD, ~1 above) and one light-difference unit. Outputs: forward drive, minus a
strong slow-down for every close sensor, plus a turn away from the close side (front
-> turn right), plus light seeking (more light on the left -> turn left).
Usage: python fear_seed.py REPO GA_NPZ OUT_NPZ [check]
"""
import sys
REPO, GENOMES, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from multiprocessing import Pool
import numpy as np

K, FRONT_THRESHOLD, SIDE_GAIN, LIGHT_GAIN = 15.0, 0.85, 5.0, 10.0
BASE, CENTRE, SLOW, TURN, SEEK, SIDE_SLOW, SIDE_THRESHOLD = 0.5, 0.3, 0.8, 0.6, 0.2, 0.4, 0.9


def fear_genome(base=BASE, centre=CENTRE, slow=SLOW, turn=TURN, seek=SEEK, side_slow=SIDE_SLOW,
                front_threshold=FRONT_THRESHOLD, side_threshold=SIDE_THRESHOLD):
    """Hidden: h0 = tanh(5 (L - R)) side difference; h1 = front danger; h2 = side danger
    (mean of L and R above side_threshold); h3 = light difference."""
    w_in = np.zeros((5, 4))
    w_in[0, 0], w_in[2, 0] = SIDE_GAIN, -SIDE_GAIN
    w_in[1, 1] = K
    w_in[0, 2] = w_in[2, 2] = K / 2
    w_in[3, 3], w_in[4, 3] = LIGHT_GAIN, -LIGHT_GAIN
    b_h = np.array([0.0, -K * front_threshold, -K * side_threshold, 0.0])
    # danger d = (h + 1) / 2 -> weight coef/2 on h and coef/2 added to the output bias
    w_out = np.zeros((4, 2))
    w_out[0] = (centre, -centre)                       # left closer -> left wheel faster -> turn right
    front = np.array([-slow + turn, -slow - turn])     # front close: slow down and turn right
    side = np.array([-side_slow, -side_slow])          # walls close at the sides: slow down
    w_out[1], w_out[2] = front / 2, side / 2
    w_out[3] = (-seek, seek)                           # more light on the left -> turn left
    b_out = base + front / 2 + side / 2
    return np.concatenate([w_in.ravel(), b_h, w_out.ravel(), b_out])


def check(spec):
    from evolution.genetic_algorithm import GeneticAlgorithm
    from evolution.metrics import measure_controller, random_maze_episodes
    name, genome = spec
    net = GeneticAlgorithm(population_size=4, seed=0, light_occlusion="blocked").decode_genome(genome)
    m = measure_controller(net, random_maze_episodes(50, seed=31, noise_seed_base=310000))["mean"]
    return name, m


if __name__ == "__main__":
    genome = fear_genome()
    np.savez(OUT, fear=genome)
    if len(sys.argv) > 4:
        g = np.load(GENOMES)
        specs = [("fear", genome), ("fear seek0.1", fear_genome(seek=0.1)), ("fear seek0.2 turn1", fear_genome(seek=0.2, turn=1.0)),
                 ("fear seek0.2 base0.7", fear_genome(seek=0.2, base=0.7)), ("seed3", g["seed3"]), ("seed2", g["seed2"])]
        with Pool(6) as p:
            for name, m in p.map(check, specs):
                print(f"{name:14} goal {m['goal_reached']:.2f} clearance {m['wall_clearance']:.3f} speed {m['speed']:.2f} "
                      f"coll% {m['pct_steps_in_collision']:.1f} bias {m['turning_bias']:+.2f} explore {m['exploration']:.2f} "
                      f"energy {m['energy_per_step']:.2f}")
