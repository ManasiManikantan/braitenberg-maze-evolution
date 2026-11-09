"""Matplotlib animation for a robot navigating a maze."""

import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from .world import World


def animate_world(world=None, steps=1000, interval=40, show=True):
    """Animate a world's maze, robot, sensor rays, goal, and movement trail.

    Args:
        world: Simulation to display. A default ``World`` is created if omitted.
        steps: Maximum number of simulation steps to animate.
        interval: Delay between frames in milliseconds.
        show: Call ``plt.show()`` before returning when true.

    Returns:
        The ``FuncAnimation`` instance, which should be retained by callers
        using a non-blocking Matplotlib backend.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if interval <= 0:
        raise ValueError("interval must be positive")
    if world is None:
        world = World()

    maze = world.maze
    vehicle = world.vehicle
    vehicle.sense(maze)

    figure, axes = plt.subplots(figsize=(8, 8))
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(maze.x_min - 0.75, maze.x_max + 0.75)
    axes.set_ylim(maze.y_min - 0.75, maze.y_max + 0.75)
    axes.set_xlabel("x (maze units)")
    axes.set_ylabel("y (maze units)")
    axes.set_title("Maze navigation")
    axes.grid(True, color="#e5e7eb", linewidth=0.6)
    axes.set_axisbelow(True)

    for (x_start, y_start), (x_end, y_end) in maze.walls:
        axes.plot(
            (x_start, x_end), (y_start, y_end),
            color="#25313c", linewidth=3.0, solid_capstyle="round",
        )

    for x_min, y_min, x_max, y_max in maze.obstacles:
        axes.add_patch(Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            facecolor="#d6dce1", edgecolor="#596570", linewidth=1.2,
        ))

    goal = Circle(
        world.goal, world.goal_radius,
        facecolor="#78b88a", edgecolor="#247245", alpha=0.35, linewidth=1.5,
    )
    axes.add_patch(goal)
    axes.plot(world.goal[0], world.goal[1], marker="*", color="#247245", markersize=12)

    trail, = axes.plot([], [], color="#d58a32", linewidth=1.8, alpha=0.9)
    robot = Circle(
        (vehicle.x_pos, vehicle.y_pos), vehicle.radius,
        facecolor="#4d91c6", edgecolor="#174c70", linewidth=1.5, zorder=5,
    )
    axes.add_patch(robot)
    heading = FancyArrowPatch(
        (0.0, 0.0), (0.0, 0.0),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.8,
        color="#174c70", zorder=6,
    )
    axes.add_patch(heading)

    left_ray, = axes.plot([], [], color="#d58a32", linewidth=1.4, zorder=4)
    right_ray, = axes.plot([], [], color="#347e9e", linewidth=1.4, zorder=4)
    status = axes.text(
        0.02, 0.98, "", transform=axes.transAxes,
        ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#ccd2d8", "alpha": 0.9},
    )

    simulated_frames = 0

    def update(frame):
        nonlocal simulated_frames
        while simulated_frames < frame:
            if not world.reached_goal:
                world.step()
            simulated_frames += 1
        vehicle.sense(maze)

        positions = world.position_history
        trail.set_data(
            [position[0] for position in positions],
            [position[1] for position in positions],
        )
        robot.center = (vehicle.x_pos, vehicle.y_pos)

        heading_end = (
            vehicle.x_pos + 1.5 * vehicle.radius * math.cos(vehicle.orientation),
            vehicle.y_pos + 1.5 * vehicle.radius * math.sin(vehicle.orientation),
        )
        heading.set_positions((vehicle.x_pos, vehicle.y_pos), heading_end)

        sensor_rays = (
            (left_ray, vehicle.ls, vehicle.left_distance, vehicle.orientation + vehicle.angle_offset),
            (right_ray, vehicle.rs, vehicle.right_distance, vehicle.orientation - vehicle.angle_offset),
        )
        for ray, sensor, distance, angle in sensor_rays:
            ray.set_data(
                (sensor.x_pos, sensor.x_pos + distance * math.cos(angle)),
                (sensor.y_pos, sensor.y_pos + distance * math.sin(angle)),
            )

        goal_status = "goal reached" if world.reached_goal else "navigating"
        status.set_text(
            f"step {world.steps} / {steps}    collisions {world.collisions}\n{goal_status}"
        )
        return trail, robot, heading, left_ray, right_ray, status

    animation = FuncAnimation(
        figure,
        update,
        frames=range(steps + 1),
        interval=interval,
        repeat=False,
        blit=False,
    )
    if show:
        plt.show()
    return animation