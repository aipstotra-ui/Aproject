"""Model I, step 2: quick lower limits on the evacuation time.

These are "it cannot possibly be faster than this" numbers, found with two simple ideas:

  Limit 1 (walking):  the last group to start moving still has to walk out.
                      time >= reaction delay + the longest of the zones' quickest walks out

  Limit 2 (doors):    everybody has to squeeze through some set of doors. If the narrowest set of doors
                      lets C people per second through, then N people need at least N / C seconds.
                      The "narrowest set of doors" is found with a max-flow / min-cut calculation,
                      which also tells us WHERE the building is throttled.

We do limit 2 for the whole audience and for parts of it (orchestra only, balcony only, ...).
"""
import networkx as nx

from bailey.params import premovement_delay

# groups of seating zones we look at separately (zone names start with O_, Z_M or B_)
ZONE_SETS = {
    "all": lambda zone: True,
    "orchestra": lambda zone: zone.startswith("O_"),
    "first_floor": lambda zone: zone.startswith("O_") or zone.startswith("Z_M"),
    "balcony": lambda zone: zone.startswith("B_"),
}


def min_cut(net, zones):
    """How many people per second can get from these zones to the exits at most (max-flow),
    and which edges form the bottleneck (min-cut). Returns (people per second, list of edge names)."""
    G = nx.DiGraph()
    for i, edge in enumerate(net.edges):
        if net.cap[i] > 1e-9:
            G.add_edge(edge["from"], edge["to"], capacity=float(net.cap[i]))

    # one pretend "source" that feeds all chosen zones, and one pretend "sink" behind all exits
    # (an edge without a 'capacity' value counts as unlimited)
    for zone in zones:
        G.add_edge("__SRC__", zone)
    for exit_name in net.exits:
        G.add_edge(exit_name, "__SINK__")

    for zone in zones:
        if not G.has_node(zone):
            return 0.0, []

    value, (source_side, sink_side) = nx.minimum_cut(G, "__SRC__", "__SINK__")

    # the bottleneck edges go from the source side to the sink side
    cut_edges = []
    for u in source_side:
        for v in G.successors(u):
            if v in sink_side:
                cut_edges.append(u + "->" + v)
    return float(value), sorted(cut_edges)


def lower_bounds(net, premovement, tol=None):
    """Both limits for one network. `premovement` is the reaction-time profile name (e.g. "concert")."""
    numerics = net.params["numerics"]
    if tol is None:
        tol = numerics["done_tolerance"]
    delay = premovement_delay(net.params, premovement)

    # ---- Limit 1: the longest "quickest walk out" of any zone, after the reaction delay ----
    paths = net.zone_paths(planning=False)
    quickest_walk = {}
    for zone in net.zones:
        if paths[zone]:
            walk_times = [net.path_tau(path) for path in paths[zone].values()]
            quickest_walk[zone] = min(walk_times)
    limit_walk = delay + max(quickest_walk.values())

    result = {
        "premovement": premovement,
        "t_min": delay,
        "t_cap": delay,
        "B1_last_starter": limit_walk,
        "max_free_travel": max(quickest_walk.values()),
        "cuts": {},
    }

    # ---- Limit 2: people / narrowest-doors capacity, for each group of zones ----
    best = limit_walk
    for group_name, belongs in ZONE_SETS.items():
        chosen_zones = [zone for zone in net.zones if belongs(zone)]
        people = float(sum(net.nodes[zone]["seats"] for zone in chosen_zones))
        people_per_second, cut_edges = min_cut(net, chosen_zones)
        if people_per_second > 0:
            limit_doors = delay + (people - tol) / people_per_second
        else:
            limit_doors = float("inf")

        door_groups = sorted({net.edges[net.eidx[name]]["group"] for name in cut_edges})
        result["cuts"][group_name] = {"people": people, "cut_capacity_pps": people_per_second,
                                      "bound": limit_doors, "cut_groups": door_groups, "cut_edges": cut_edges}
        best = max(best, limit_doors)

    result["T_lower_bound"] = best       # the strongest (largest) of all the limits
    return result
