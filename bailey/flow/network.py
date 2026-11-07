"""Model I network: building.json + params.yaml -> arrays used by the flow models.

An Evacuation network is a directed graph whose edges carry
  cap[e]   maximum flow, persons/s (SFPE: Fsm x effective width)
  tau[e]   free travel time, s (length / free speed)
and whose nodes carry a storage limit (area x jam density) used for queue spill-back.

Obstructions: `blocked` maps edge id -> remaining capacity fraction (0 = fully blocked).
If `known` is True the obstruction is known when routes are planned (Crowd Managers redirect at t=0);
otherwise routes are planned on the intact building and people discover the obstruction on arrival.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from bailey.params import BUILDING_PATH, capacity, free_speed, load_params


def load_building(path=None) -> dict:
    with open(path or BUILDING_PATH) as f:
        return json.load(f)


@dataclass
class Network:
    building: dict
    params: dict
    mode: str = "emergency"                  # "emergency": all exits; "normal": only edges with use == "normal"
    blocked: dict[str, float] = field(default_factory=dict)
    known: bool = True

    def __post_init__(self) -> None:
        P = self.params
        self.node_ids = [n["id"] for n in self.building["nodes"]]
        self.nidx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.nodes = {n["id"]: n for n in self.building["nodes"]}
        self.exits = [x for x in self.building["exits"]]
        self.zones = [n["id"] for n in self.building["nodes"] if n["kind"] == "zone"]
        self.seats = np.array([self.nodes[z]["seats"] for z in self.zones], dtype=float)
        self.N = float(self.seats.sum())

        edges = [e for e in self.building["edges"] if self.mode == "emergency" or e["use"] == "normal"]
        for eid in self.blocked:
            if not any(e["id"] == eid for e in edges):
                raise KeyError(f"blocked edge {eid!r} not in network")
        self.edges = edges
        self.eidx = {e["id"]: i for i, e in enumerate(edges)}
        self.tail = np.array([self.nidx[e["from"]] for e in edges])
        self.head = np.array([self.nidx[e["to"]] for e in edges])
        self.cap_intact = np.array([capacity(P, e["kind"], e["width"], e["n"]) for e in edges])
        factor = np.array([self.blocked.get(e["id"], 1.0) for e in edges])
        self.cap = self.cap_intact * factor
        self.tau = np.array([e["length"] / free_speed(P, e["kind"]) for e in edges])
        dmax = P["hydraulic"]["d_max"]
        self.storage = np.array([
            (self.nodes[n]["area"] * dmax) if self.nodes[n]["area"] else math.inf for n in self.node_ids
        ])
        self.is_exit = np.array([self.nodes[n]["kind"] == "exit" for n in self.node_ids])

    # ------------------------------------------------------------------ graphs
    def graph(self, *, planning: bool = True) -> nx.DiGraph:
        """DiGraph weighted by free travel time. With planning=True and an unknown obstruction,
        the obstruction is invisible (routes are planned on the intact building)."""
        G = nx.DiGraph()
        G.add_nodes_from(self.node_ids)
        caps = self.cap_intact if (planning and not self.known) else self.cap
        for i, e in enumerate(self.edges):
            if caps[i] > 1e-9:
                G.add_edge(e["from"], e["to"], idx=i, tau=self.tau[i], cap=caps[i])
        return G

    def path_edges(self, nodes: list[str]) -> list[int]:
        return [self.eidx[f"{a}->{b}"] for a, b in zip(nodes[:-1], nodes[1:])]

    def path_tau(self, path: list[int]) -> float:
        return float(self.tau[path].sum())

    def zone_paths(self, *, planning: bool = True) -> dict[str, dict[str, list[int]]]:
        """For every zone and every reachable exit: the fastest (free-flow) path, as edge indices."""
        G = self.graph(planning=planning)
        out: dict[str, dict[str, list[int]]] = {}
        for z in self.zones:
            dist, paths = nx.single_source_dijkstra(G, z, weight="tau")
            out[z] = {x: self.path_edges(paths[x]) for x in self.exits if x in paths}
        return out

    def exit_of(self, path: list[int]) -> str:
        return self.node_ids[self.head[path[-1]]]

    def summary_table(self) -> list[dict]:
        rows = []
        for i, e in enumerate(self.edges):
            rows.append({"edge": e["id"], "kind": e["kind"], "group": e["group"], "use": e["use"],
                         "width_m": e["width"], "n": e["n"], "length_m": e["length"],
                         "cap_pps": round(float(self.cap[i]), 3), "tau_s": round(float(self.tau[i]), 2)})
        return rows

    def group_capacity(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for i, e in enumerate(self.edges):
            if e["group"] in ("corridor", "seating", "mezz_cross_aisle"):
                continue
            out[e["group"]] = out.get(e["group"], 0.0) + float(self.cap[i])
        return out


def make_network(mode: str = "emergency", blocked: dict[str, float] | None = None, known: bool = True,
                 building: dict | None = None, params: dict | None = None) -> Network:
    return Network(building or load_building(), params or load_params(), mode=mode,
                   blocked=dict(blocked or {}), known=known)


def one_leaf_blocked(building: dict, edge_id: str) -> dict[str, float]:
    """Obstruction of a single door leaf: an n-leaf edge keeps (n-1)/n of its capacity."""
    e = next(e for e in building["edges"] if e["id"] == edge_id)
    return {edge_id: (e["n"] - 1) / e["n"]}
