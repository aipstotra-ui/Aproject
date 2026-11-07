"""Multi-commodity fluid queueing simulation (Model I, level 2).

A *commodity* is a group of people who share an origin zone and a planned path (list of edges).
Each edge behaves as a capacity-limited bottleneck at its entrance (FIFO point queue held at the edge's
tail node) followed by a fixed transit delay. Queued people of different commodities are served in
proportion to their share of the queue (fluid FIFO). Optional spill-back: a node cannot hold more than
its storage (area x jam density); upstream edges are throttled when the downstream node is full.

Unknown obstructions: if a commodity's next edge has zero capacity, its people pause for a discovery
delay, walk back along their own path if they are in a dead end, and continue on the fastest open
path (a virtual 'retreat' edge carries the delay plus the walk-back time).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from bailey.flow.network import Network
from bailey.flow.quickest_lp import transit_steps


@dataclass
class Commodity:
    zone: str
    path: list[int]          # edge indices (into the extended edge arrays)
    exit: str
    reroute_of: int = -1     # index of the commodity this one was spawned from (unknown obstruction)


@dataclass
class PathSet:
    """All candidate commodities for a network: up to k fast paths per (zone, exit)."""
    net: Network
    commodities: list[Commodity]
    by_zone: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def build(cls, net: Network, k_per_exit: int = 3, slack: float = 20.0) -> "PathSet":
        G = net.graph(planning=True)
        comms: list[Commodity] = []
        by_zone: dict[str, list[int]] = {}
        for z in net.zones:
            by_zone[z] = []
            for x in net.exits:
                if not nx.has_path(G, z, x):
                    continue
                best = None
                for i, nodes in enumerate(nx.shortest_simple_paths(G, z, x, weight="tau")):
                    p = net.path_edges(nodes)
                    t = net.path_tau(p)
                    best = t if best is None else best
                    if i >= k_per_exit or t > best + slack:
                        break
                    by_zone[z].append(len(comms))
                    comms.append(Commodity(z, p, x))
        return cls(net, comms, by_zone)

    def tau(self, k: int) -> float:
        return self.net.path_tau(self.commodities[k].path)

    def fastest(self, zone: str, exits: list[str] | None = None) -> int:
        ks = [k for k in self.by_zone[zone] if exits is None or self.commodities[k].exit in exits]
        return min(ks, key=self.tau)


@dataclass
class FluidResult:
    T: float
    cumulative: np.ndarray            # persons out by end of each step
    exit_totals: dict[str, float]
    comm_out: np.ndarray              # persons of each (extended) commodity that got out
    comm_mean_exit: np.ndarray        # mean exit time per (extended) commodity
    queue_hist: np.ndarray            # (E_ext, H) queue length at each edge entrance
    edge_ids: list[str]
    mean_exit_time: float
    dt: float

    def in_building(self, N: float) -> np.ndarray:
        return N - self.cumulative


def simulate(net: Network, paths: PathSet, frac: np.ndarray, release: np.ndarray, dt: float,
             horizon: float, *, spillback: bool = True, tol: float = 0.5,
             discovery_delay: float = 10.0) -> FluidResult:
    """Run the fluid model. `frac[k]` = share of zone(k)'s audience that plans to use commodity k."""
    H = int(np.ceil(horizon / dt))
    rel = np.zeros(H)
    rel[: min(H, len(release))] = release[:H]

    # ---------------- extended edge set (+ hesitation edges) and commodities (+ reroutes)
    E = len(net.edges)
    tail, head = list(net.tail), list(net.head)
    cap, steps = list(net.cap), list(transit_steps(net, dt))
    edge_ids = [e["id"] for e in net.edges]
    comms = list(paths.commodities)
    w = list(frac)
    actual = net.graph(planning=False)
    virtual_edges: dict[tuple[int, int, int], int] = {}

    def retreat_edge(v: int, u: int, delay: float) -> int:
        """Virtual edge v -> u: pause to realise the way is blocked, then walk back to u (u == v: no walk)."""
        n_steps = max(1, int(round(delay / dt)))
        key = (v, u, n_steps)
        if key not in virtual_edges:
            virtual_edges[key] = len(tail)
            tail.append(v); head.append(u); cap.append(np.inf); steps.append(n_steps)
            edge_ids.append(f"hesitate@{net.node_ids[v]}" + (f"->back@{net.node_ids[u]}" if u != v else ""))
        return virtual_edges[key]

    trans = []      # (edge, comm) -> (next edge, next comm)
    exits = []      # (edge, comm, exit name)
    first = []      # first edge of each commodity (after rerouting at the zone itself)
    reroute_cache: dict[tuple[int, int], int] = {}

    def reroute(parent: int, i: int) -> int:
        """Commodity `parent` stands at the tail of its blocked i-th edge. Pause, walk back along its own
        path as far as needed to reach a node with an open exit, then take the fastest open path."""
        key = (parent, i)
        if key in reroute_cache:
            return reroute_cache[key]
        p = comms[parent].path
        v = tail[p[i]]
        back = 0.0
        for j in range(i, -1, -1):
            u = tail[p[j]]
            if j < i:
                back += net.tau[p[j]] if p[j] < E else 0.0
            dist, pths = nx.single_source_dijkstra(actual, net.node_ids[u], weight="tau")
            reachable = [x for x in net.exits if x in pths]
            if reachable:
                break
        else:
            raise RuntimeError(f"no open exit reachable from {net.node_ids[v]} or anywhere behind it")
        x = min(reachable, key=lambda x: dist[x])
        new_path = [retreat_edge(v, u, discovery_delay + back)] + net.path_edges(pths[x])
        reroute_cache[key] = len(comms)
        comms.append(Commodity(comms[parent].zone, new_path, x, reroute_of=parent))
        w.append(0.0)
        return reroute_cache[key]

    k = 0
    while k < len(comms):  # comms grows while iterating (reroutes)
        p = comms[k].path
        if cap[p[0]] <= 1e-9:  # zone's own first edge blocked: discover at the seat
            kk = reroute(k, 0)
            first.append((comms[kk].path[0], kk))
        else:
            first.append((p[0], k))
        for i, e in enumerate(p):
            if i + 1 == len(p):
                exits.append((e, k, comms[k].exit))
            elif cap[p[i + 1]] <= 1e-9:
                kk = reroute(k, i + 1)
                trans.append((e, k, comms[kk].path[0], kk))
            else:
                trans.append((e, k, p[i + 1], k))
        k += 1

    Ex, K = len(tail), len(comms)
    tail_a, head_a = np.array(tail), np.array(head)
    cap_a = np.array(cap, dtype=float)
    steps_a = np.array(steps)
    w_a = np.array(w)
    zone_seats = {z: net.seats[i] for i, z in enumerate(net.zones)}
    rel_amount = np.array([zone_seats[c.zone] for c in comms]) * w_a   # persons per commodity
    fe = np.array([f[0] for f in first]); fk = np.array([f[1] for f in first])
    te, tk, ne, nk = (np.array(a, dtype=int) for a in zip(*trans)) if trans else (np.zeros(0, int),) * 4
    xe = np.array([x[0] for x in exits]); xk = np.array([x[1] for x in exits])
    x_name = [x[2] for x in exits]
    exit_names = list(net.exits)
    x_col = np.array([exit_names.index(n) for n in x_name])

    storage = net.storage.copy()
    if not spillback:
        storage[:] = np.inf
    finite_store = np.isfinite(storage) & ~net.is_exit
    V = len(net.node_ids)

    L = int(steps_a.max()) + 1
    Q = np.zeros((Ex, K))
    buf = np.zeros((L, Ex, K))
    transit = np.zeros(Ex)
    cum = np.zeros(H)
    exit_tot = np.zeros(len(exit_names))
    comm_out = np.zeros(K)
    comm_time = np.zeros(K)
    qhist = np.zeros((Ex, H))
    out_total = 0.0
    T = np.inf
    # release goes to the commodity's first edge (only original commodities have weight)
    for t in range(H):
        # 1. pre-movement release
        if rel[t] > 0:
            np.add.at(Q, (fe, fk), rel_amount[: len(fe)] * rel[t])
        # 2. arrivals at edge heads
        slot = t % L
        A = buf[slot]
        if A.any():
            transit -= A.sum(1)
            np.add.at(Q, (ne, nk), A[te, tk])
            got = A[xe, xk]
            np.add.at(exit_tot, x_col, got)
            np.add.at(comm_out, xk, got)
            np.add.at(comm_time, xk, got * t * dt)
            out_total += got.sum()
            buf[slot] = 0.0
        cum[t] = out_total
        if out_total >= net.N - tol and not np.isfinite(T):
            T = t * dt
            qhist = qhist[:, : t + 1]
            break
        # 3. discharge into edges
        q = Q.sum(1)
        qhist[:, t] = q
        d = np.minimum(cap_a * dt, q)
        if finite_store.any():
            content = np.bincount(tail_a, weights=q - d, minlength=V) + np.bincount(head_a, weights=transit, minlength=V)
            req = np.bincount(head_a, weights=d, minlength=V)
            space = np.maximum(storage - content, 0.0)
            scale = np.ones(V)
            m = finite_store & (req > space)
            scale[m] = space[m] / req[m]
            d = d * scale[head_a]
        pos = q > 1e-12
        D = np.zeros_like(Q)
        D[pos] = Q[pos] * (d[pos] / q[pos])[:, None]
        Q -= D
        buf[(t + steps_a) % L, np.arange(Ex)] += D
        transit += d

    mean_exit = np.divide(comm_time, comm_out, out=np.full(K, np.nan), where=comm_out > 1e-9)
    return FluidResult(T=float(T), cumulative=cum[: (int(T / dt) + 1 if np.isfinite(T) else H)],
                       exit_totals=dict(zip(exit_names, exit_tot.tolist())), comm_out=comm_out,
                       comm_mean_exit=mean_exit, queue_hist=qhist, edge_ids=edge_ids,
                       mean_exit_time=float(comm_time.sum() / max(comm_out.sum(), 1e-9)), dt=dt)
