# Walls hurt v2: scraping costs health too

The first walls-hurt world (`../danger-worlds`) charged health only for new
contacts, so robots learned to scrape along walls for free.

**The new rule.** `World(health=15, contact_step_cost=0.1)`:
- 1 health per new hit;
- plus 0.1 for every step spent touching a wall.

**Calibration** (`scripts/calibrate_contact.py`, 50 mazes):

| robot | survival |
| --- | --- |
| left wall follower | 78% |
| right wall follower | 70% |
| GA winners 3 and 2 | 42-44% |
| fear robot | 44% |

**Runs.** Seeds 3 and 2 plus the fear robot twice. Each run: 20 generations,
parent + 29 mutated copies.

**Comparison** (`scripts/measure_hurt2.py`). Every robot is measured in the
same plain world on 50 mazes, plus deaths in the v2 world. The groups are:
- the parents;
- the zoo's safe control;
- the loophole version, v1;
- v2.

Outputs: `comparison.csv`, `report.txt`, `careful_comparison.png`,
`fear_clearance_v2_vs_safe.png`.

## Results

**GA lineages (seeds 3 and 2).**
- Wall contact is back to the safe-control level: 13-17% of steps, against
  22-26% under v1.
- Clearance and speed match the safe control, and deaths fell only slightly.
- So the loophole behaviour is gone, but the robots are no more careful.

**Fear lineage: caution survives in v2 and fades in the safe world.**
- The clearance of each generation's best ends at 0.16-0.18 in v2, against
  0.11-0.12 in the safe runs.
- The final v2 robots stay slow: speed 0.32, against 0.39-0.48 in the safe runs.
- The cost: 52-56% goal success in the plain world, against 72-78% for the
  safe-world fear robots.
- Wall contact stays high (28-39%). The fear robot is slow and scrapes along
  side walls that its +/-60 degree sensors do not see.
