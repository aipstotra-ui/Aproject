"""Model I, step 1: turn the building description into a network the computer can work with.

The building (data/building.json) is a list of
  * nodes: places, e.g. a seating zone, a door, a piece of corridor, an exit
  * edges: one-way connections between two places, e.g. "mezzanine door 4 -> corridor"

For every edge we work out two numbers:
  * capacity    how many people per second can pass (from the SFPE door/stair flow rates)
  * walk time   how many seconds it takes to walk along it at normal speed

For every node we work out how many people can stand there before it is full (its storage).

A blocked door is handled by lowering its capacity (to 0 if it is completely blocked).
"""
import json
import math

import networkx as nx
import numpy as np

from bailey.params import BUILDING_PATH, capacity, free_speed, load_params


def load_building(path=None):
    """Read the building description (data/building.json) into a Python dictionary."""
    if path is None:
        path = BUILDING_PATH
    with open(path) as f:
        return json.load(f)


class Network:
    """The building as a network of places (nodes) and connections (edges).

    mode     "emergency" = every exit may be used; "normal" = only the everyday routes
    blocked  {edge id: how much of its capacity is left}, e.g. {"C090->FOYER": 0.0}
    known    True  = everybody knows about the blocked door from the start
             False = people only find out when they get there
    """

    def __init__(self, building, params, mode="emergency", blocked=None, known=True):
        self.building = building
        self.params = params
        self.mode = mode
        self.blocked = dict(blocked or {})
        self.known = known

        # ---- Step 1: the places (nodes) ----
        self.node_ids = []       # list of node names, e.g. ["C000", "C010", ...]
        self.nidx = {}           # node name -> its position in that list
        self.nodes = {}          # node name -> all information about it
        for node in building["nodes"]:
            self.nidx[node["id"]] = len(self.node_ids)
            self.node_ids.append(node["id"])
            self.nodes[node["id"]] = node

        self.exits = list(building["exits"])

        # seating zones and how many people sit in each
        self.zones = []
        seat_counts = []
        for node in building["nodes"]:
            if node["kind"] == "zone":
                self.zones.append(node["id"])
                seat_counts.append(node["seats"])
        self.seats = np.array(seat_counts, dtype=float)
        self.N = float(sum(seat_counts))          # total number of people (1,329)

        # ---- Step 2: the connections (edges) ----
        # In a normal departure people only use the everyday routes.
        self.edges = []
        for edge in building["edges"]:
            if mode == "emergency" or edge["use"] == "normal":
                self.edges.append(edge)

        # check that every blocked door really exists
        edge_names = [edge["id"] for edge in self.edges]
        for name in self.blocked:
            if name not in edge_names:
                raise KeyError(f"blocked edge {name!r} not in network")

        self.eidx = {}           # edge name -> its position in self.edges
        for i, edge in enumerate(self.edges):
            self.eidx[edge["id"]] = i

        # where each edge starts (tail) and ends (head), as node positions
        self.tail = np.array([self.nidx[edge["from"]] for edge in self.edges])
        self.head = np.array([self.nidx[edge["to"]] for edge in self.edges])

        # ---- Step 3: capacity and walking time of every edge ----
        capacity_list = []
        capacity_after_blocking = []
        walk_time_list = []
        for edge in self.edges:
            full_capacity = capacity(params, edge["kind"], edge["width"], edge["n"])
            share_left = self.blocked.get(edge["id"], 1.0)      # 1.0 = not blocked
            capacity_list.append(full_capacity)
            capacity_after_blocking.append(full_capacity * share_left)
            walk_time_list.append(edge["length"] / free_speed(params, edge["kind"]))
        self.cap_intact = np.array(capacity_list)      # capacity if nothing were blocked
        self.cap = np.array(capacity_after_blocking)   # capacity actually available
        self.tau = np.array(walk_time_list)            # walking time in seconds

        # ---- Step 4: how many people fit in each place ----
        # storage = floor area x the most people that fit on one square metre (3.8)
        max_density = params["hydraulic"]["d_max"]
        storage_list = []
        exit_flags = []
        for name in self.node_ids:
            node = self.nodes[name]
            if node["area"]:
                storage_list.append(node["area"] * max_density)
            else:
                storage_list.append(math.inf)          # seats and exits: no limit
            exit_flags.append(node["kind"] == "exit")
        self.storage = np.array(storage_list)
        self.is_exit = np.array(exit_flags)

    # ------------------------------------------------------------------ helpers
    def graph(self, planning=True):
        """Make a networkx graph for finding shortest paths. Each edge's 'tau' is its walking time.

        If a blocked door is not known yet (known=False), people plan their route as if the door
        were open, so with planning=True we leave the blocked door in the graph."""
        if planning and not self.known:
            caps = self.cap_intact
        else:
            caps = self.cap
        G = nx.DiGraph()
        G.add_nodes_from(self.node_ids)
        for i, edge in enumerate(self.edges):
            if caps[i] > 1e-9:                         # a fully blocked edge is left out
                G.add_edge(edge["from"], edge["to"], idx=i, tau=self.tau[i], cap=caps[i])
        return G

    def path_edges(self, node_names):
        """Turn a list of places [A, B, C] into the list of edge positions [A->B, B->C]."""
        edge_list = []
        for i in range(len(node_names) - 1):
            edge_list.append(self.eidx[node_names[i] + "->" + node_names[i + 1]])
        return edge_list

    def path_tau(self, edge_list):
        """Total walking time of a path (list of edge positions)."""
        total = 0.0
        for e in edge_list:
            total += self.tau[e]
        return float(total)

    def zone_paths(self, planning=True):
        """For every seating zone and every exit it can reach: the quickest path (edge positions)."""
        G = self.graph(planning=planning)
        result = {}
        for zone in self.zones:
            distances, paths = nx.single_source_dijkstra(G, zone, weight="tau")
            result[zone] = {}
            for exit_name in self.exits:
                if exit_name in paths:
                    result[zone][exit_name] = self.path_edges(paths[exit_name])
        return result

    def exit_of(self, edge_list):
        """The exit at the end of a path."""
        return self.node_ids[self.head[edge_list[-1]]]

    def summary_table(self):
        """One row per edge with its size, capacity and walking time (for the paper's tables)."""
        rows = []
        for i, edge in enumerate(self.edges):
            rows.append({"edge": edge["id"], "kind": edge["kind"], "group": edge["group"], "use": edge["use"],
                         "width_m": edge["width"], "n": edge["n"], "length_m": edge["length"],
                         "cap_pps": round(float(self.cap[i]), 3), "tau_s": round(float(self.tau[i]), 2)})
        return rows

    def group_capacity(self):
        """Total capacity of each group of doors/stairs, e.g. all 9 mezzanine doors together."""
        totals = {}
        for i, edge in enumerate(self.edges):
            if edge["group"] in ("corridor", "seating", "mezz_cross_aisle"):
                continue
            totals[edge["group"]] = totals.get(edge["group"], 0.0) + float(self.cap[i])
        return totals


def make_network(mode="emergency", blocked=None, known=True, building=None, params=None):
    """Shortcut: load the building and the settings, then build the network."""
    if building is None:
        building = load_building()
    if params is None:
        params = load_params()
    return Network(building, params, mode=mode, blocked=blocked, known=known)


def one_leaf_blocked(building, edge_id):
    """Block one door leaf. A door made of n leaves keeps (n-1)/n of its capacity;
    a single door (n = 1) is blocked completely."""
    for edge in building["edges"]:
        if edge["id"] == edge_id:
            return {edge_id: (edge["n"] - 1) / edge["n"]}
    raise KeyError(edge_id)
