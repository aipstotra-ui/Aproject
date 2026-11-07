"""Static floor fields for Model II: walking time [s] from every cell to every exit.

Cost of a step u -> v = length / (v_free * mean speed factor of u, v) * (1 + beta * rho_v), where rho_v
is the local density around v (fraction of occupied cells in its neighbourhood). With beta = 0 this is
the classical static field (shortest walking time). With beta > 0 it becomes congestion-aware, so agents
prefer a free door over a jammed one. This is the CA counterpart of Model I's tactical equilibrium.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from bailey.sim.grid import EXITS, CellGrid


class FieldSolver:
    def __init__(self, grid: CellGrid, v_free: float, blocked: set[str] = frozenset()):
        self.grid = grid
        E, L = grid.edges(blocked)
        # directed both ways
        self.src = np.concatenate([E[:, 0], E[:, 1]])
        self.dst = np.concatenate([E[:, 1], E[:, 0]])
        sp_mean = 0.5 * (grid.speed[self.src] + grid.speed[self.dst])
        self.base = np.concatenate([L, L]) / (v_free * sp_mean)
        n = grid.n
        self.adj = sp.csr_matrix((np.ones(len(self.src)), (self.src, self.dst)), shape=(n, n))
        self.deg = np.asarray(self.adj.sum(1)).ravel()
        self.exit_cells = [grid.exit_cells[e] for e in EXITS]

    def density(self, occ: np.ndarray) -> np.ndarray:
        return (occ + self.adj @ occ) / (1.0 + self.deg)

    def solve(self, occ: np.ndarray | None = None, beta: float = 0.0) -> np.ndarray:
        """(n_exits, n_cells) time-to-go; inf where an exit is unreachable."""
        w = self.base
        if occ is not None and beta > 0:
            w = self.base * (1.0 + beta * self.density(occ.astype(float))[self.dst])
        # reversed graph: edge v -> u carries the cost of stepping u -> v
        M = sp.csr_matrix((w, (self.dst, self.src)), shape=(self.grid.n, self.grid.n))
        out = np.empty((len(EXITS), self.grid.n))
        for i, cells in enumerate(self.exit_cells):
            out[i] = dijkstra(M, directed=True, indices=cells, min_only=True) if len(cells) else np.inf
        return out
