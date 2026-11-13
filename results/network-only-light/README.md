# Network-only evolution with goal-light sensors and a maze-size curriculum

5-input network (3 distance sensors at -60/0/+60 degrees + 2 light sensors at
+/-45 degrees). Trained on 3 fresh random mazes per evaluation with a
curriculum (12x12 for generations 0-19, 16x16 for 20-39, 20x20 for 40-59),
per-maze limit 2 x shortest route at 0.05 units/step, outcome-only fitness,
default robot noise (incl. light noise 0.02). 5 seeds x 60 generations x
population 30 per occlusion mode:

- `blocked`: walls block the line of sight from a light sensor to the goal.
- `through`: walls do not block the light.

Final controller per seed chosen on a fixed 20-maze validation set.

`summary.csv` compares, on the same 50 random test mazes and courses 1-4 as
`../network-only-random-mazes`: evolved light controllers, untrained 5-input
networks (first genome of each seed's initial population), and the previous
no-light random-maze controllers.

`light_wiring.csv`: mean partial derivatives of each motor output with
respect to each light input over states visited on 20 test mazes, with the
resulting turn direction and Braitenberg-equivalent behaviour. Note the
simulator convention: heading += turn_gain * (left - right), so a larger
LEFT motor turns the robot LEFT (the opposite of a physical differential
drive). Same-side excitatory wiring in code therefore turns toward the light
here (aggression-like), as crossed excitatory wiring would on a real robot.
