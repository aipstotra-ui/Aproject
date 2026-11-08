"""Model II, step 2: the "time to exit" map (called a floor field).

For every cell and every exit we compute: how many seconds would it take to walk from this cell to
that exit? People then walk "downhill" on this map.

Cost of one step from cell u to neighbour v:
    step length / (walking speed)  x  (1 + beta x how crowded v is)
  * walking speed = normal speed x the average speed factor of u and v (slower in seats and on stairs)
  * "how crowded" = share of occupied cells among v and its neighbours (0 = empty, 1 = packed)
With beta = 0 this is just the walking time. With beta > 0 a crowded doorway "feels" farther away,
so people drift to a free door instead.

The shortest total cost from every cell to an exit is found with Dijkstra's algorithm (from scipy).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from bailey.sim.grid import EXITS


class FieldSolver:
    def __init__(self, grid, v_free, blocked=frozenset()):
        self.grid = grid
        pairs, lengths = grid.edges(blocked)

        # every pair can be walked in both directions: list each direction separately
        self.src = np.concatenate([pairs[:, 0], pairs[:, 1]])     # step from ...
        self.dst = np.concatenate([pairs[:, 1], pairs[:, 0]])     # ... to
        step_length = np.concatenate([lengths, lengths])
        average_speed = 0.5 * (grid.speed[self.src] + grid.speed[self.dst])
        self.base = step_length / (v_free * average_speed)        # seconds per step, no crowding

        # neighbour table as a matrix: adj[u, v] = 1 if you can step from u to v
        n = grid.n
        self.adj = sp.csr_matrix((np.ones(len(self.src)), (self.src, self.dst)), shape=(n, n))
        self.deg = np.asarray(self.adj.sum(axis=1)).ravel()      # number of neighbours of each cell
        self.exit_cells = [grid.exit_cells[exit_name] for exit_name in EXITS]

    def density(self, occupied):
        """Share of occupied cells around each cell (the cell itself + its neighbours)."""
        occupied_neighbours = self.adj @ occupied
        return (occupied + occupied_neighbours) / (1.0 + self.deg)

    def solve(self, occupied=None, beta=0.0):
        """Time to each exit from every cell: an array [exit, cell]. inf = that exit cannot be reached."""
        step_cost = self.base
        if occupied is not None and beta > 0:
            crowding = self.density(occupied.astype(float))
            step_cost = self.base * (1.0 + beta * crowding[self.dst])

        # Dijkstra goes OUT from its start cells, but we want the distance TO the exit.
        # So we turn every step around: the step u -> v is stored as v -> u, then start at the exit.
        reversed_steps = sp.csr_matrix((step_cost, (self.dst, self.src)), shape=(self.grid.n, self.grid.n))
        times = np.empty((len(EXITS), self.grid.n))
        for number, cells in enumerate(self.exit_cells):
            if len(cells) == 0:
                times[number] = np.inf
            else:
                times[number] = dijkstra(reversed_steps, directed=True, indices=cells, min_only=True)
        return times
