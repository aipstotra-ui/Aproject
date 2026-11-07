"""Routing policies for Model I: how each zone's audience splits over candidate paths.

Each policy returns `frac` with frac[k] = share of zone(k)'s people who plan to use commodity k
(shares sum to 1 within every zone).

Route choice has two levels:
  * strategic  -- WHICH EXIT a person heads for (habit / nearest / assigned / free choice);
  * tactical   -- WHICH DOORS AND CORRIDOR DIRECTION they take to reach it. People see local queues,
                  so the tactical split is an equilibrium among the paths to the chosen exit.
A *class* is (zone, allowed commodities, share). `equilibrate` finds the tactical user equilibrium of
all classes jointly (method of successive averages, MSA).

  habit     exit = OUT_FRONT, the way they came in
  nearest   exit = nearest exit by free walking time (ties split evenly)
  ue        any exit: fully informed selfish choice (user equilibrium)
  assigned  static system-optimal assignment from an LP (an idealised Crowd-Manager choreography or
            barrier layout); paths are fixed by the LP, no tactical adaptation
  mix       population shares of the above, e.g. {"assigned": 0.6, "habit": 0.25, "nearest": 0.15}
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.stats import lognorm

from bailey.flow.fluid import PathSet, simulate
from bailey.flow.network import Network


Class = tuple[str, list[int], float]


def classes_habit(paths: PathSet, share: float = 1.0) -> list[Class]:
    out = []
    for z in paths.net.zones:
        ks = [k for k in paths.by_zone[z] if paths.commodities[k].exit == "OUT_FRONT"]
        out.append((z, ks or list(paths.by_zone[z]), share))
    return out


def classes_nearest(paths: PathSet, share: float = 1.0, tie: float = 2.0) -> list[Class]:
    out = []
    for z in paths.net.zones:
        best = {}
        for k in paths.by_zone[z]:
            x = paths.commodities[k].exit
            best[x] = min(best.get(x, np.inf), paths.tau(k))
        tmin = min(best.values())
        near = [x for x, t in best.items() if t <= tmin + tie]
        for x in near:
            out.append((z, [k for k in paths.by_zone[z] if paths.commodities[k].exit == x], share / len(near)))
    return out


def classes_free(paths: PathSet, share: float = 1.0) -> list[Class]:
    return [(z, list(paths.by_zone[z]), share) for z in paths.net.zones]


def classes_fixed_exit(paths: PathSet, exit_share: dict[str, dict[str, float]], share: float = 1.0) -> list[Class]:
    """Explicit exit assignment per zone, e.g. from a barrier layout or LED plan: {zone: {exit: share}}."""
    out = []
    for z, xs in exit_share.items():
        for x, sh in xs.items():
            ks = [k for k in paths.by_zone[z] if paths.commodities[k].exit == x]
            if sh > 0 and ks:
                out.append((z, ks, share * sh))
    return out


def assigned(paths: PathSet, premovement: dict, n_grid: int = 25, q_bulk: float = 0.95) -> tuple[np.ndarray, float]:
    """Static system-optimal assignment (LP). Returns (frac, LP bound on makespan).

    For a routing x (persons per commodity), the load on edge e is l_e = sum_{k ∋ e} x_k.
    People reach e no earlier than (pre-movement + a_e), where a_e is the fastest free-flow time from
    any origin to e. With pre-movement CDF F, the queue at e cannot clear before
        a_e + s + l_e (1 - F(s)) / c_e      for every s,
    and the last person still needs tau_e + b_e to get out. Hence, for every e and grid value s,
        T >= a_e + s + tau_e + b_e + l_e (1 - F(s)) / c_e,
    plus T >= t_cap + (fastest path of zone z). Minimise T, then (tie-break) total travel time.

    The s-grid stops at the q_bulk quantile of pre-movement: beyond it the constraint would force every
    lightly used path to carry a fluid "last straggler", which the simulation's tolerance ignores.
    The LP value is therefore an *approximation*, not a bound; the rigorous lower bound is T* from
    quickest_lp.solve_quickest.
    """
    net = paths.net
    K = len(paths.commodities)
    dist = lognorm(s=premovement["sigma"], scale=premovement["median"])
    t_cap = float(dist.ppf(premovement["q_cap"]))
    s_max = float(dist.ppf(q_bulk))
    s_grid = np.linspace(premovement["t_min"], s_max, n_grid)
    F0 = dist.cdf(premovement["t_min"])
    F = np.clip((dist.cdf(s_grid) - F0) / (dist.cdf(t_cap) - F0), 0, 1)

    E = len(net.edges)
    a = np.full(E, np.inf); b = np.full(E, np.inf)
    inc = np.zeros((E, K))
    for k, c in enumerate(paths.commodities):
        cum = np.concatenate([[0.0], np.cumsum(net.tau[c.path])])
        total = cum[-1]
        for i, e in enumerate(c.path):
            inc[e, k] = 1.0
            a[e] = min(a[e], cum[i])
            b[e] = min(b[e], total - cum[i + 1])
    used = np.where(inc.any(1) & np.isfinite(net.cap) & (net.cap > 1e-9))[0]

    # variables: x_0..x_{K-1}, T
    rows, rhs = [], []
    for e in used:
        for s, Fs in zip(s_grid, F):
            row = np.zeros(K + 1)
            row[:K] = inc[e] * (1 - Fs) / net.cap[e]
            row[K] = -1.0
            rows.append(row); rhs.append(-(a[e] + s + net.tau[e] + b[e]))
    for z in net.zones:  # last starter
        row = np.zeros(K + 1); row[K] = -1.0
        rows.append(row); rhs.append(-(t_cap + min(paths.tau(k) for k in paths.by_zone[z])))
    A_ub = np.array(rows); b_ub = np.array(rhs)
    A_eq = np.zeros((len(net.zones), K + 1)); b_eq = np.zeros(len(net.zones))
    for i, z in enumerate(net.zones):
        A_eq[i, paths.by_zone[z]] = 1.0
        b_eq[i] = net.seats[i]
    c1 = np.zeros(K + 1); c1[K] = 1.0
    r1 = linprog(c1, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (K + 1), method="highs")
    if r1.status != 0:
        raise RuntimeError(r1.message)
    T_lp = r1.x[K]
    # stage 2: among (near-)optimal routings, minimise total walking time
    c2 = np.zeros(K + 1); c2[:K] = [paths.tau(k) for k in range(K)]
    r2 = linprog(c2, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=[(0, None)] * K + [(0, T_lp * (1 + 1e-4) + 1e-6)], method="highs")
    x = (r2.x if r2.status == 0 else r1.x)[:K]
    frac = np.zeros(K)
    for i, z in enumerate(net.zones):
        ks = paths.by_zone[z]
        frac[ks] = x[ks] / net.seats[i]
    return frac, float(T_lp)


def path_times_given_queues(paths: PathSet, res, dt: float, t0_grid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Expected exit time of every commodity's planned path for someone starting at t0 ~ weights,
    given the queue history of a fluid run (wait at edge e = queue / capacity)."""
    net = paths.net
    E = len(net.edges)
    qh = res.queue_hist[:E]
    Hh = qh.shape[1]
    cap = net.cap
    tau = net.tau
    out = np.zeros(len(paths.commodities))
    for k, c in enumerate(paths.commodities):
        t = t0_grid.copy()
        for e in c.path:
            idx = np.minimum((t / dt).astype(int), Hh - 1)
            wait = np.where(cap[e] > 1e-9, qh[e, idx] / np.maximum(cap[e], 1e-9), np.inf)
            t = t + wait + max(1, round(tau[e] / dt)) * dt
        out[k] = float((t * weights).sum())
    return out


def equilibrate(paths: PathSet, classes: list[Class], release: np.ndarray, dt: float, horizon: float,
                iters: int = 25, fixed: np.ndarray | None = None, **sim_kw) -> tuple[np.ndarray, list[float]]:
    """Tactical user equilibrium by MSA. `fixed` = extra frac that does not adapt (e.g. LP-assigned share)."""
    net = paths.net
    base = np.zeros(len(paths.commodities)) if fixed is None else fixed
    cls_frac = [np.zeros(len(paths.commodities)) for _ in classes]
    for c, (z, ks, sh) in zip(cls_frac, classes):
        c[min(ks, key=paths.tau)] = sh
    t_grid = np.arange(len(release)) * dt
    sel = release > 1e-6
    t0, wts = t_grid[sel], release[sel] / release[sel].sum()
    history = []
    frac = base + sum(cls_frac)
    if not classes:
        return frac, history
    for n in range(1, iters + 1):
        res = simulate(net, paths, frac, release, dt, horizon, **sim_kw)
        history.append(res.T)
        times = path_times_given_queues(paths, res, dt, t0, wts)
        for i, (z, ks, sh) in enumerate(classes):
            target = np.zeros_like(frac)
            target[ks[int(np.argmin(times[ks]))]] = sh
            cls_frac[i] = (1 - 1.0 / (n + 1)) * cls_frac[i] + (1.0 / (n + 1)) * target
        frac = base + sum(cls_frac)
    return frac, history


def build_policy(paths: PathSet, spec: dict[str, float], release: np.ndarray, dt: float, horizon: float,
                 premovement: dict, iters: int = 25, exit_plan: dict | None = None, **sim_kw) -> np.ndarray:
    """Population mix -> frac. spec keys: habit, nearest, ue, assigned, plan (needs exit_plan)."""
    tot = sum(spec.values())
    classes: list[Class] = []
    fixed = np.zeros(len(paths.commodities))
    for name, wt in spec.items():
        sh = wt / tot
        if sh <= 0:
            continue
        if name == "habit":
            classes += classes_habit(paths, sh)
        elif name == "nearest":
            classes += classes_nearest(paths, sh)
        elif name == "ue":
            classes += classes_free(paths, sh)
        elif name == "plan":
            classes += classes_fixed_exit(paths, exit_plan, sh)
        elif name == "assigned":
            fixed += sh * assigned(paths, premovement)[0]
        else:
            raise ValueError(name)
    frac, _ = equilibrate(paths, classes, release, dt, horizon, iters=iters, fixed=fixed, **sim_kw)
    return frac
