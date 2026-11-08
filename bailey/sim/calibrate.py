"""Calibration of the CA against the SFPE hydraulic model.

A rectangular room (width x depth) drains through a k-cell door into an exit row. With everyone
starting at once the door saturates; the steady discharge per door cell is compared with the SFPE
maximum specific flow Fsm x cell (= 1.316 p/s/m x 0.5 m = 0.658 p/s per cell for level doors).
"""
from __future__ import annotations

import math

import numpy as np

from bailey.sim.engine import CAConfig, Engine
from bailey.sim.grid import EXITS, Aperture, CellGrid


def toy_room(width_cells: int = 20, depth_cells: int = 20, door_cells: int = 3, h: float = 0.5) -> CellGrid:
    xs, ys, reg = [], [], []
    idx = {}
    for i in range(width_cells):
        for j in range(depth_cells + 1):          # j == depth_cells is the exit row
            idx[(i, j)] = len(xs)
            xs.append((i + 0.5) * h); ys.append((j + 0.5) * h)
            reg.append("OUT_FRONT" if j == depth_cells else "room")
    base, blen, door = [], [], []
    lo = width_cells // 2 - door_cells // 2
    for (i, j), c in idx.items():
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            o = idx.get((i + di, j + dj))
            if o is None:
                continue
            if reg[c] == reg[o]:
                base.append((c, o)); blen.append(h * math.hypot(di, dj))
            elif lo <= i < lo + door_cells and lo <= i + di < lo + door_cells:
                door.append((c, o))
    x, y = np.array(xs), np.array(ys)
    reg = np.array(reg, dtype=object)
    exit_cells = {e: np.array([], int) for e in EXITS}
    exit_cells["OUT_FRONT"] = np.where(reg == "OUT_FRONT")[0]
    ap = Aperture("door", door, (0, width_cells * h / 2, depth_cells * h), door_cells,
                  [math.hypot(x[a] - x[b], y[a] - y[b]) for a, b in door])
    return CellGrid(h=h, x=x, y=y, layer=np.zeros(len(x), int), region=reg, speed=np.ones(len(x)),
                    base_edges=np.array(base), base_len=np.array(blen), apertures={"door": ap},
                    exit_cells=exit_cells, zone_cells={"room": np.where(reg == "room")[0]},
                    plot_xy=np.column_stack([x, y]))


def door_flow(params: dict, door_cells: int = 3, people: int = 250, seeds: int = 5) -> float:
    """Steady discharge [persons/s per door cell], measured between the 20th and 80th percentile exits."""
    g = toy_room(door_cells=door_cells)
    eng = Engine(g, params)
    pm = {"delay": 0.0}
    flows = []
    for s in range(seeds):
        r = eng.run(CAConfig(routing={"nearest": 1.0}, premovement=pm), seed=s, seats={"room": people})
        et = np.sort(r.exit_time[np.isfinite(r.exit_time)])
        a, b = int(0.2 * len(et)), int(0.8 * len(et))
        flows.append((b - a) / (et[b] - et[a]))
    return float(np.mean(flows)) / door_cells
