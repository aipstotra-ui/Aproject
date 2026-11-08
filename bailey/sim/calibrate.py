"""Model II, step 5: check (and tune) how fast people get through a door.

We build a small test room, 10 m x 10 m, with one door on one side and a row of exit cells behind it.
300 people start at once, so a queue forms at the door straight away. We record when each person gets
out and measure the flow in the busy middle part (between the 20 % and 80 % mark).

Fire engineers (SFPE Handbook) say a busy door lets through 1.316 people per second per metre of width,
which is 0.658 people per second for each 0.5 m cell of door. We tune the bumping chance (friction mu)
until our door gives the same number.
"""
import math

import numpy as np

from bailey.sim.engine import CAConfig, Engine
from bailey.sim.grid import EXITS, Aperture, CellGrid


def toy_room(width_cells=20, depth_cells=20, door_cells=3, h=0.5):
    """A square room (width x depth cells) with a door of `door_cells` cells in the middle of one wall."""
    xs, ys, regions = [], [], []
    cell_at = {}
    for i in range(width_cells):
        for j in range(depth_cells + 1):          # the last row (j == depth_cells) is the exit row
            cell_at[(i, j)] = len(xs)
            xs.append((i + 0.5) * h)
            ys.append((j + 0.5) * h)
            if j == depth_cells:
                regions.append("OUT_FRONT")
            else:
                regions.append("room")

    pairs, lengths, door_pairs = [], [], []
    door_start = width_cells // 2 - door_cells // 2       # first column of the door
    for (i, j), cell in cell_at.items():
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):  # each pair of neighbours once
            other = cell_at.get((i + di, j + dj))
            if other is None:
                continue
            if regions[cell] == regions[other]:
                pairs.append((cell, other))
                lengths.append(h * math.hypot(di, dj))
            elif door_start <= i < door_start + door_cells and door_start <= i + di < door_start + door_cells:
                door_pairs.append((cell, other))          # room -> exit row, only inside the door

    x = np.array(xs)
    y = np.array(ys)
    regions = np.array(regions, dtype=object)
    exit_cells = {exit_name: np.array([], int) for exit_name in EXITS}
    exit_cells["OUT_FRONT"] = np.where(regions == "OUT_FRONT")[0]
    door_lengths = [math.hypot(x[a] - x[b], y[a] - y[b]) for a, b in door_pairs]
    door = Aperture("door", door_pairs, (0, width_cells * h / 2, depth_cells * h), door_cells, door_lengths)
    return CellGrid(h=h, x=x, y=y, layer=np.zeros(len(x), int), region=regions, speed=np.ones(len(x)),
                    base_edges=np.array(pairs), base_len=np.array(lengths), apertures={"door": door},
                    exit_cells=exit_cells, zone_cells={"room": np.where(regions == "room")[0]},
                    plot_xy=np.column_stack([x, y]))


def door_flow(params, door_cells=3, people=250, seeds=5):
    """Average flow through the test door, in people per second per door cell."""
    room = toy_room(door_cells=door_cells)
    engine = Engine(room, params)
    settings = CAConfig(routing={"nearest": 1.0}, premovement={"delay": 0.0})
    flows = []
    for seed in range(seeds):
        result = engine.run(settings, seed=seed, seats={"room": people})
        exit_times = np.sort(result.exit_time[np.isfinite(result.exit_time)])
        start = int(0.2 * len(exit_times))
        end = int(0.8 * len(exit_times))
        people_counted = end - start
        seconds = exit_times[end] - exit_times[start]
        flows.append(people_counted / seconds)
    return float(np.mean(flows)) / door_cells
