"""Quickest evacuation flow on a time-expanded network (Model I, level 1).

Discretise time in steps of dt. Edge e has capacity cap[e]*dt persons per step and transit time
s_e = max(1, round(tau_e/dt)) steps. Zone z releases N_z * rel[t] people at step t (pre-movement).
Nodes may hold people (queues) up to their storage limit.

Variables  f[e,t] >= 0  persons entering edge e at step t      (<= cap_e dt)
           h[v,t] >= 0  persons waiting at node v from t to t+1 (<= storage_v)
Flow conservation at every non-exit node and step:
    supply[v,t] + sum_{e->v} f[e, t - s_e] + h[v, t-1] = sum_{v->e} f[e,t] + h[v,t]
Objective: maximise sum over arrivals at exits of (H - arrival step), i.e. minimise total person-time in
the building. With a single (super-)sink an *earliest-arrival* flow exists (Minieka 1973; Baumann &
Skutella 2009), so this solution also minimises the makespan T*: no routing whatsoever, under the
same capacities and pre-movement, empties the building faster. T* is the "system optimum" benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from bailey.flow.network import Network


def transit_steps(net: Network, dt: float) -> np.ndarray:
    return np.maximum(1, np.rint(net.tau / dt)).astype(int)


@dataclass
class QuickestResult:
    T: float                     # makespan [s] (inf if not everyone is out by the horizon)
    cumulative: np.ndarray       # persons out by end of each step
    exit_totals: dict[str, float]
    edge_totals: np.ndarray      # total persons through each edge
    total_person_time: float     # sum of evacuation times [person*s]
    status: str


def solve_quickest(net: Network, release: np.ndarray, dt: float, horizon: float,
                   tol: float = 0.5) -> QuickestResult:
    H = int(np.ceil(horizon / dt))
    rel = np.zeros(H)
    rel[: min(H, len(release))] = release[:H]
    E, V = len(net.edges), len(net.node_ids)
    s = transit_steps(net, dt)
    usable = net.cap > 1e-9

    nf = E * H
    fidx = lambda e, t: e * H + t  # noqa: E731
    hidx = lambda v, t: nf + v * H + t  # noqa: E731
    nvar = nf + V * H

    # bounds
    ub = np.empty(nvar)
    ub[:nf] = np.repeat(np.where(usable, net.cap * dt, 0.0), H)
    tt = np.tile(np.arange(H), E)
    ee = np.repeat(np.arange(E), H)
    ub[:nf][tt + s[ee] > H - 1] = 0.0  # arrivals after the horizon are not allowed
    store = np.where(np.isinf(net.storage), np.inf, net.storage)
    store[net.is_exit] = 0.0
    ub[nf:] = np.repeat(store, H)
    ub = np.where(np.isinf(ub), None, ub)  # linprog: None = unbounded
    bounds = list(zip(np.zeros(nvar), ub))

    # objective
    c = np.full(nvar, 1e-4)  # tiny cost on movement: no pointless circulation
    into_exit = net.is_exit[net.head]
    arr_step = tt + s[ee]
    c[:nf] -= np.where(into_exit[ee], H - arr_step, 0.0)
    c[nf:] = 0.0

    # conservation rows for non-exit nodes
    rows, cols, vals = [], [], []
    nonexit = np.where(~net.is_exit)[0]
    rowof = -np.ones(V, dtype=int)
    rowof[nonexit] = np.arange(len(nonexit))
    T = np.arange(H)
    for e in range(E):
        u, w = net.tail[e], net.head[e]
        if rowof[u] >= 0:  # outflow of u at t
            rows.append(rowof[u] * H + T); cols.append(fidx(e, T)); vals.append(np.ones(H))
        if rowof[w] >= 0:  # inflow to w at t + s_e
            t0 = T[T + s[e] < H]
            rows.append(rowof[w] * H + t0 + s[e]); cols.append(fidx(e, t0)); vals.append(-np.ones(len(t0)))
    for v in nonexit:
        r = rowof[v] * H
        rows.append(r + T); cols.append(hidx(v, T)); vals.append(np.ones(H))
        rows.append(r + T[1:]); cols.append(hidx(v, T[:-1])); vals.append(-np.ones(H - 1))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(nonexit) * H, nvar))
    b = np.zeros(len(nonexit) * H)
    for zi, z in enumerate(net.zones):
        v = net.nidx[z]
        b[rowof[v] * H: rowof[v] * H + H] = net.seats[zi] * rel

    res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"quickest-flow LP failed: {res.message}")
    f = res.x[:nf].reshape(E, H)

    arrivals = np.zeros(H)
    exit_totals: dict[str, float] = {x: 0.0 for x in net.exits}
    for e in np.where(into_exit)[0]:
        t0 = T[T + s[e] < H]
        np.add.at(arrivals, t0 + s[e], f[e, t0])
        exit_totals[net.node_ids[net.head[e]]] += float(f[e].sum())
    cum = np.cumsum(arrivals)
    done = np.where(cum >= net.N - tol)[0]
    makespan = float(done[0] * dt) if len(done) else float("inf")
    return QuickestResult(T=makespan, cumulative=cum, exit_totals=exit_totals, edge_totals=f.sum(1),
                          total_person_time=float((np.arange(H) * dt * arrivals).sum()), status=res.message)
