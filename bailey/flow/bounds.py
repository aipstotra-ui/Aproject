"""Analytic lower bounds on the egress time (Model I, level 0).

For any routing, the time T at which the last person leaves satisfies
  (B1) T >= max_z [ s_z + tau_z ]          s_z = time zone z starts moving (the fixed reaction delay);
                                           tau_z = its fastest free-flow path to any exit
  (B2) T >= t_min + N_S / C_S   for every set of zones S,
       where C_S is the min-cut capacity (persons/s) separating S from all exits.
The binding cut identifies *where* the building is throttled (e.g. the nine mezzanine doors).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from bailey.flow.network import Network
from bailey.params import premovement_release, premovement_stats

ZONE_SETS = {
    "all": lambda z: True,
    "orchestra": lambda z: z.startswith("O_"),
    "first_floor": lambda z: z.startswith("O_") or z.startswith("Z_M"),
    "balcony": lambda z: z.startswith("B_"),
}


def min_cut(net: Network, zones: list[str]) -> tuple[float, list[str]]:
    """Max-flow / min-cut from the given zones to all exits. Returns (capacity p/s, cut edge ids)."""
    G = nx.DiGraph()
    for i, e in enumerate(net.edges):
        if net.cap[i] > 1e-9:
            G.add_edge(e["from"], e["to"], capacity=float(net.cap[i]))
    for z in zones:
        G.add_edge("__SRC__", z)            # no capacity attribute = infinite
    for x in net.exits:
        G.add_edge(x, "__SINK__")
    if not all(G.has_node(z) for z in zones):
        return 0.0, []
    value, (S, T) = nx.minimum_cut(G, "__SRC__", "__SINK__")
    cut = [f"{u}->{v}" for u in S for v in G.successors(u) if v in T]
    return float(value), sorted(cut)


def lower_bounds(net: Network, premovement: str, tol: float | None = None) -> dict:
    pm = premovement_stats(net.params, premovement)
    num = net.params["numerics"]
    tol = num["done_tolerance"] if tol is None else tol
    dt = num["dt"]
    cum = np.cumsum(premovement_release(net.params, premovement, dt, num["horizon"]))
    paths = net.zone_paths(planning=False)
    tau_z = {z: min(net.path_tau(p) for p in paths[z].values()) for z in net.zones if paths[z]}
    b1 = 0.0
    for z, tz in tau_z.items():
        n_z = net.nodes[z]["seats"]
        s_z = int(np.argmax(n_z * (1 - cum) <= tol)) * dt   # first step after which < tol remain unreleased
        b1 = max(b1, s_z + tz)
    out = {"premovement": premovement, "t_min": pm["t_min"], "t_cap": pm["t_cap"],
           "B1_last_starter": b1, "max_free_travel": max(tau_z.values()), "cuts": {}}
    best = b1
    for name, pred in ZONE_SETS.items():
        zs = [z for z in net.zones if pred(z)]
        n_s = float(sum(net.nodes[z]["seats"] for z in zs))
        c, cut = min_cut(net, zs)
        b2 = pm["t_min"] + ((n_s - tol) / c if c > 0 else float("inf"))
        groups = sorted({net.edges[net.eidx[eid]]["group"] for eid in cut})
        out["cuts"][name] = {"people": n_s, "cut_capacity_pps": c, "bound": b2, "cut_groups": groups,
                             "cut_edges": cut}
        best = max(best, b2)
    out["T_lower_bound"] = best
    return out
