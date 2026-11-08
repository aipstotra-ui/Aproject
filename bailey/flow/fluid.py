"""Model I, step 4: simulate the crowd like water flowing through pipes, one second at a time.

Words used here:
  * route group ("commodity"): all people from one seating zone who plan to take the same path
    (a list of edges) to the same exit.
  * queue: people standing at the start of an edge, waiting to get in (e.g. in front of a door).
  * in transit: people who got into an edge and are walking along it.

Every second we do three things:
  1. RELEASE   people who finish reacting join the queue of the first edge of their path.
  2. ARRIVE    people who finished walking along an edge join the queue of their next edge
               (or leave the building, if that edge ended at an exit).
  3. ENTER     from each queue, as many people as the edge's capacity allows walk into the edge.
               Different route groups in one queue get in in proportion to how many of them wait there
               (fair first-come-first-served for a fluid).
               Spill-back: if the place at the end of an edge is already full, fewer people may enter.

Blocked door that people don't know about: when a route group reaches it, they pause (discovery delay),
walk back along their own path if they are in a dead end, and take the quickest open way out. We model
this as a new route group that starts with a pretend "pause and walk back" edge.
"""
import networkx as nx
import numpy as np

from bailey.flow.quickest_lp import transit_steps


class Commodity:
    """One route group: people from `zone` taking `path` (list of edge positions) to `exit`."""

    def __init__(self, zone, path, exit, reroute_of=-1):
        self.zone = zone
        self.path = path
        self.exit = exit
        self.reroute_of = reroute_of        # which route group this one was made from (blocked door)


class PathSet:
    """All route options: for every zone and exit, up to 3 quick paths (within 20 s of the quickest)."""

    def __init__(self, net, commodities, by_zone):
        self.net = net
        self.commodities = commodities      # list of Commodity
        self.by_zone = by_zone              # zone -> list of positions in `commodities`

    @classmethod
    def build(cls, net, k_per_exit=3, slack=20.0):
        G = net.graph(planning=True)
        commodities = []
        by_zone = {}
        for zone in net.zones:
            by_zone[zone] = []
            for exit_name in net.exits:
                if not nx.has_path(G, zone, exit_name):
                    continue
                # networkx lists paths from quickest to slowest
                quickest = None
                for number, node_names in enumerate(nx.shortest_simple_paths(G, zone, exit_name, weight="tau")):
                    path = net.path_edges(node_names)
                    walk = net.path_tau(path)
                    if quickest is None:
                        quickest = walk
                    if number >= k_per_exit or walk > quickest + slack:
                        break
                    by_zone[zone].append(len(commodities))
                    commodities.append(Commodity(zone, path, exit_name))
        return cls(net, commodities, by_zone)

    def tau(self, k):
        """Walking time of route group k's path."""
        return self.net.path_tau(self.commodities[k].path)

    def fastest(self, zone, exits=None):
        """The route group with the shortest walk from `zone` (optionally only to some exits)."""
        options = []
        for k in self.by_zone[zone]:
            if exits is None or self.commodities[k].exit in exits:
                options.append(k)
        return min(options, key=self.tau)


class FluidResult:
    """What simulate() returns."""

    def __init__(self, T, cumulative, exit_totals, comm_out, comm_mean_exit, queue_hist, edge_ids,
                 mean_exit_time, dt):
        self.T = T                            # time the building is empty [s]
        self.cumulative = cumulative          # people out by the end of each step
        self.exit_totals = exit_totals        # people through each exit
        self.comm_out = comm_out              # people of each route group who got out
        self.comm_mean_exit = comm_mean_exit  # average exit time of each route group
        self.queue_hist = queue_hist          # queue length at each edge, every step
        self.edge_ids = edge_ids              # names of the edges (incl. pretend "pause" edges)
        self.mean_exit_time = mean_exit_time  # average exit time of everybody
        self.dt = dt

    def in_building(self, N):
        """People still inside at each step."""
        return N - self.cumulative


def simulate(net, paths, frac, release, dt, horizon, spillback=True, tol=0.5, discovery_delay=10.0):
    """Run the fluid simulation.

    frac[k]     share of zone(k)'s people who plan to use route group k
    release[t]  share of every zone that starts moving at step t
    """
    num_steps = int(np.ceil(horizon / dt))
    release_per_step = np.zeros(num_steps)
    for t in range(min(num_steps, len(release))):
        release_per_step[t] = release[t]

    # =====================================================================================
    # PART A: set-up (lists we fill once, before the clock starts)
    # =====================================================================================
    num_real_edges = len(net.edges)
    edge_start = list(net.tail)                 # place where each edge starts
    edge_end = list(net.head)                   # place where each edge ends
    edge_capacity = list(net.cap)               # people per second
    edge_steps = list(transit_steps(net, dt))   # walking time in steps
    edge_ids = [edge["id"] for edge in net.edges]
    groups = list(paths.commodities)            # route groups (more may be added for blocked doors)
    group_share = list(frac)                    # share of the zone using each group
    real_graph = net.graph(planning=False)      # the building as it really is (blocked doors removed)

    # --- A1. pretend "pause and walk back" edges, used when someone finds a blocked door ---
    pause_edge_of = {}

    def pause_edge(at_place, back_to_place, seconds):
        steps = max(1, int(round(seconds / dt)))
        key = (at_place, back_to_place, steps)
        if key not in pause_edge_of:
            pause_edge_of[key] = len(edge_start)
            edge_start.append(at_place)
            edge_end.append(back_to_place)
            edge_capacity.append(np.inf)        # no capacity limit on a pause
            edge_steps.append(steps)
            name = "hesitate@" + net.node_ids[at_place]
            if back_to_place != at_place:
                name += "->back@" + net.node_ids[back_to_place]
            edge_ids.append(name)
        return pause_edge_of[key]

    # --- A2. a new route group for people who find edge i of group `parent`'s path blocked ---
    new_group_of = {}

    def reroute(parent, i):
        if (parent, i) in new_group_of:
            return new_group_of[(parent, i)]
        path = groups[parent].path
        stuck_at = edge_start[path[i]]
        # walk back along their own path until some open exit can be reached
        walk_back = 0.0
        found = False
        for j in range(i, -1, -1):
            go_back_to = edge_start[path[j]]
            if j < i and path[j] < num_real_edges:
                walk_back += net.tau[path[j]]
            distances, routes = nx.single_source_dijkstra(real_graph, net.node_ids[go_back_to], weight="tau")
            open_exits = [exit_name for exit_name in net.exits if exit_name in routes]
            if open_exits:
                found = True
                break
        if not found:
            raise RuntimeError(f"no open exit reachable from {net.node_ids[stuck_at]} or anywhere behind it")
        best_exit = min(open_exits, key=lambda exit_name: distances[exit_name])
        new_path = [pause_edge(stuck_at, go_back_to, discovery_delay + walk_back)]
        new_path += net.path_edges(routes[best_exit])
        new_group_of[(parent, i)] = len(groups)
        groups.append(Commodity(groups[parent].zone, new_path, best_exit, reroute_of=parent))
        group_share.append(0.0)                 # nobody starts in this group; people are moved into it
        return new_group_of[(parent, i)]

    # --- A3. for every group: its first edge, and where people go after each edge ---
    first_edge = []          # (edge, group) where the group's people start queueing
    next_step = []           # (edge, group, next edge, next group) after walking along an edge
    leave_at = []            # (edge, group, exit name) edges that end at an exit
    k = 0
    while k < len(groups):   # (the list can grow while we go: reroute() adds groups)
        path = groups[k].path
        if edge_capacity[path[0]] <= 1e-9:
            # the very first edge is blocked: people notice it at their seat
            new_k = reroute(k, 0)
            first_edge.append((groups[new_k].path[0], new_k))
        else:
            first_edge.append((path[0], k))
        for i in range(len(path)):
            edge = path[i]
            if i == len(path) - 1:
                leave_at.append((edge, k, groups[k].exit))
            elif edge_capacity[path[i + 1]] <= 1e-9:
                new_k = reroute(k, i + 1)       # next edge blocked -> switch to the new group
                next_step.append((edge, k, groups[new_k].path[0], new_k))
            else:
                next_step.append((edge, k, path[i + 1], k))
        k += 1

    num_edges = len(edge_start)
    num_groups = len(groups)

    # how many people each group releases in total
    seats_of_zone = {}
    for zone_number, zone in enumerate(net.zones):
        seats_of_zone[zone] = net.seats[zone_number]
    people_in_group = np.array([seats_of_zone[g.zone] * group_share[k] for k, g in enumerate(groups)])

    # turn the lists into numpy arrays (needed for speed inside the clock loop)
    edge_start = np.array(edge_start)
    edge_end = np.array(edge_end)
    edge_capacity = np.array(edge_capacity, dtype=float)
    edge_steps = np.array(edge_steps)
    first_e = np.array([pair[0] for pair in first_edge])
    first_g = np.array([pair[1] for pair in first_edge])
    from_e = np.array([row[0] for row in next_step], dtype=int)
    from_g = np.array([row[1] for row in next_step], dtype=int)
    to_e = np.array([row[2] for row in next_step], dtype=int)
    to_g = np.array([row[3] for row in next_step], dtype=int)
    exit_e = np.array([row[0] for row in leave_at])
    exit_g = np.array([row[1] for row in leave_at])
    exit_names = list(net.exits)
    exit_number = np.array([exit_names.index(row[2]) for row in leave_at])

    storage = net.storage.copy()
    if not spillback:
        storage[:] = np.inf
    has_limit = np.isfinite(storage) & ~net.is_exit
    num_places = len(net.node_ids)

    # --- A4. the state of the simulation ---
    # waiting[e, g]                  people of group g in the queue of edge e
    # walking[(t mod L), e, g]       people of group g who will reach the end of edge e at step t
    # walking_total[e]               people currently walking along edge e
    longest = int(edge_steps.max()) + 1
    waiting = np.zeros((num_edges, num_groups))
    walking = np.zeros((longest, num_edges, num_groups))
    walking_total = np.zeros(num_edges)

    people_out_by_step = np.zeros(num_steps)
    out_per_exit = np.zeros(len(exit_names))
    out_per_group = np.zeros(num_groups)
    exit_time_sum_per_group = np.zeros(num_groups)
    queue_history = np.zeros((num_edges, num_steps))
    people_out = 0.0
    T = np.inf

    # =====================================================================================
    # PART B: the clock
    # =====================================================================================
    for t in range(num_steps):

        # ---- 1. RELEASE: people who finish reacting join their first queue ----
        if release_per_step[t] > 0:
            np.add.at(waiting, (first_e, first_g), people_in_group[: len(first_e)] * release_per_step[t])

        # ---- 2. ARRIVE: people reaching the end of an edge ----
        slot = t % longest
        arriving = walking[slot]                          # [edge, group]
        if arriving.any():
            walking_total -= arriving.sum(axis=1)
            # they join the queue of their next edge
            np.add.at(waiting, (to_e, to_g), arriving[from_e, from_g])
            # or they are out of the building
            got_out = arriving[exit_e, exit_g]
            np.add.at(out_per_exit, exit_number, got_out)
            np.add.at(out_per_group, exit_g, got_out)
            np.add.at(exit_time_sum_per_group, exit_g, got_out * t * dt)
            people_out += got_out.sum()
            walking[slot] = 0.0

        people_out_by_step[t] = people_out
        if people_out >= net.N - tol:                     # (almost) nobody left inside
            T = t * dt
            queue_history = queue_history[:, : t + 1]
            break

        # ---- 3. ENTER: from each queue, up to capacity walk into the edge ----
        queue = waiting.sum(axis=1)                       # total queue at each edge
        queue_history[:, t] = queue
        entering = np.minimum(edge_capacity * dt, queue)

        # spill-back: the place at the end of an edge cannot take more people than it has room for
        if has_limit.any():
            # people at each place: queues there (after this step's departures) + walking towards it
            people_at_place = (np.bincount(edge_start, weights=queue - entering, minlength=num_places)
                               + np.bincount(edge_end, weights=walking_total, minlength=num_places))
            wanting_in = np.bincount(edge_end, weights=entering, minlength=num_places)
            room = np.maximum(storage - people_at_place, 0.0)
            shrink = np.ones(num_places)
            too_full = has_limit & (wanting_in > room)
            shrink[too_full] = room[too_full] / wanting_in[too_full]
            entering = entering * shrink[edge_end]

        # share the places in the edge fairly between the groups in its queue
        has_queue = queue > 1e-12
        moving = np.zeros_like(waiting)
        moving[has_queue] = waiting[has_queue] * (entering[has_queue] / queue[has_queue])[:, None]
        waiting -= moving
        # they will reach the end of the edge after its walking time
        arrive_slot = (t + edge_steps) % longest
        walking[arrive_slot, np.arange(num_edges)] += moving
        walking_total += entering

    # =====================================================================================
    # PART C: summary
    # =====================================================================================
    mean_exit_per_group = np.full(num_groups, np.nan)
    for g in range(num_groups):
        if out_per_group[g] > 1e-9:
            mean_exit_per_group[g] = exit_time_sum_per_group[g] / out_per_group[g]
    if np.isfinite(T):
        people_out_by_step = people_out_by_step[: int(T / dt) + 1]
    mean_exit = float(exit_time_sum_per_group.sum() / max(out_per_group.sum(), 1e-9))
    return FluidResult(T=float(T), cumulative=people_out_by_step,
                       exit_totals=dict(zip(exit_names, out_per_exit.tolist())), comm_out=out_per_group,
                       comm_mean_exit=mean_exit_per_group, queue_hist=queue_history, edge_ids=edge_ids,
                       mean_exit_time=mean_exit, dt=dt)
