"""Model II agent decision rules: strategic exit choice for each behavioural type.

Types (same vocabulary as Model I's strategic classes, so the two models are comparable):
  HABIT     leave the way you came in (main foyer exit), unless you know it cannot be reached
  NEAREST   nearest exit by walking time (static field, ignores queues)
  GUIDED    follow a Crowd Manager / guidance plan: exit drawn from the plan's shares for your zone
  RATIONAL  minimise (walking time to exit + people ahead of you heading there / exit capacity),
            re-evaluated periodically with hysteresis ("partial knowledge": you only know the queues
            of agents already moving, not those still seated)

All functions operate on arrays of agent indices so the CA engine stays vectorised.
"""
from __future__ import annotations

import numpy as np

HABIT, NEAREST, GUIDED, RATIONAL = 0, 1, 2, 3
TYPE_OF = {"habit": HABIT, "nearest": NEAREST, "assigned": GUIDED, "plan": GUIDED, "ue": RATIONAL}
TYPE_NAMES = {HABIT: "habit", NEAREST: "nearest", GUIDED: "guided", RATIONAL: "rational"}
FRONT = 0  # index of OUT_FRONT in grid.EXITS


def nearest_exit(F_static: np.ndarray, cells: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """F_static: (n_exits, n_cells); allowed: (n_exits,) bool mask of exits the agent believes open."""
    vals = np.where(allowed[:, None], F_static[:, cells], np.inf)
    return np.argmin(vals, axis=0)


def habit_exit(F_static: np.ndarray, cells: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    ok = allowed[FRONT] & np.isfinite(F_static[FRONT, cells])
    return np.where(ok, FRONT, nearest_exit(F_static, cells, allowed))


def guided_exit(rng, shares: np.ndarray, zones: np.ndarray, allowed: np.ndarray,
                fallback: np.ndarray) -> np.ndarray:
    """shares: (n_zones, n_exits) plan shares; renormalised over exits believed open."""
    s = shares[zones] * allowed[None, :]
    tot = s.sum(1)
    out = fallback.copy()
    ok = tot > 1e-12
    if ok.any():
        cum = np.cumsum(s[ok] / tot[ok, None], axis=1)
        u = rng.random(ok.sum())[:, None]
        out[ok] = np.minimum((u > cum).sum(1), s.shape[1] - 1)
    return out


def rational_cost(F_dyn: np.ndarray, cells: np.ndarray, ahead: np.ndarray, cap: np.ndarray,
                  allowed: np.ndarray) -> np.ndarray:
    """(n_exits, n_agents) predicted time to get out."""
    cost = F_dyn[:, cells] + ahead / cap[:, None]
    return np.where(allowed[:, None], cost, np.inf)


def people_ahead(F_dyn: np.ndarray, pos: np.ndarray, target: np.ndarray, moving: np.ndarray,
                 cells: np.ndarray) -> np.ndarray:
    """For each exit e and each evaluating agent: number of moving agents targeting e who are
    closer to e (by the dynamic field) than the evaluating agent would be."""
    n_ex = F_dyn.shape[0]
    out = np.zeros((n_ex, len(cells)))
    for e in range(n_ex):
        vals = np.sort(F_dyn[e, pos[moving & (target == e)]])
        out[e] = np.searchsorted(vals, F_dyn[e, cells])
    return out
