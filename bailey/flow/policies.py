"""Model I, step 5: how people choose their route.

Every zone's people are split over the route groups (paths) available to them. The result is a list
`frac`, where frac[k] = share of zone(k)'s people who use route group k (the shares of one zone add up to 1).

People choose in two stages:
  1. WHICH EXIT (their "type"):
       habit     the main foyer (the way they came in)
       nearest   the exit with the shortest walk (if two are equally close, half go to each)
       ue        any exit: fully informed, picks whatever is quickest given everybody's queues
       assigned  the exit the Crowd Managers send them to (the best plan for the whole building)
       plan      an exit fixed in advance per zone (e.g. by barriers or LED signs)
  2. WHICH DOORS / CORRIDOR DIRECTION to that exit: people look at the queues and avoid long ones.
     We find this by trial and error (simulate, see which path was quickest, move a few people there,
     repeat). This is called the "method of successive averages".

A "class" below is (zone, list of route groups this class may use, share of the zone's people).
"""
import numpy as np
from scipy.optimize import linprog

from bailey.flow.fluid import simulate


# --------------------------------------------------------------------- stage 1: which exit
def classes_habit(paths, share=1.0):
    """Everyone heads for the main foyer exit (OUT_FRONT)."""
    classes = []
    for zone in paths.net.zones:
        options = [k for k in paths.by_zone[zone] if paths.commodities[k].exit == "OUT_FRONT"]
        if not options:                       # the foyer is unreachable: any exit
            options = list(paths.by_zone[zone])
        classes.append((zone, options, share))
    return classes


def classes_nearest(paths, share=1.0, tie=2.0):
    """Everyone heads for the exit with the shortest walk. Exits within `tie` seconds count as equal."""
    classes = []
    for zone in paths.net.zones:
        # shortest walk to each exit
        best_walk = {}
        for k in paths.by_zone[zone]:
            exit_name = paths.commodities[k].exit
            walk = paths.tau(k)
            if exit_name not in best_walk or walk < best_walk[exit_name]:
                best_walk[exit_name] = walk
        shortest = min(best_walk.values())
        nearest_exits = [exit_name for exit_name, walk in best_walk.items() if walk <= shortest + tie]
        # split the zone evenly between the (equally) nearest exits
        for exit_name in nearest_exits:
            options = [k for k in paths.by_zone[zone] if paths.commodities[k].exit == exit_name]
            classes.append((zone, options, share / len(nearest_exits)))
    return classes


def classes_free(paths, share=1.0):
    """Fully informed: any route group of the zone may be chosen."""
    classes = []
    for zone in paths.net.zones:
        classes.append((zone, list(paths.by_zone[zone]), share))
    return classes


def classes_fixed_exit(paths, exit_share, share=1.0):
    """A fixed plan: exit_share = {zone: {exit: share of the zone}}."""
    classes = []
    for zone, shares in exit_share.items():
        for exit_name, zone_share in shares.items():
            options = [k for k in paths.by_zone[zone] if paths.commodities[k].exit == exit_name]
            if zone_share > 0 and options:
                classes.append((zone, options, share * zone_share))
    return classes


# --------------------------------------------------------------------- the Crowd Managers' plan
def assigned(paths, premovement):
    """The best split of every zone over route groups, for the building as a whole (a linear program).

    For each edge e used by a plan:
      * the first people reach it no earlier than  delay + a_e   (a_e = quickest walk from any seat to e)
      * load_e people need load_e / capacity_e seconds to get through
      * the last of them still needs tau_e + b_e to walk out (b_e = quickest walk from e to an exit)
    so the time the building is empty, T, must be at least
        delay + a_e + load_e / capacity_e + tau_e + b_e     for every edge e,
        delay + quickest walk out of zone z                   for every zone z.
    The computer chooses the split that makes T as small as possible. If several splits give the same
    T, it then picks the one with the least total walking.

    Returns (frac, T of the plan).
    """
    net = paths.net
    num_groups = len(paths.commodities)
    delay = float(premovement["delay"])
    num_edges = len(net.edges)

    # ---- a_e, b_e and which groups use which edge ----
    walk_to = [np.inf] * num_edges       # a_e
    walk_from = [np.inf] * num_edges     # b_e
    uses = np.zeros((num_edges, num_groups))   # uses[e, k] = 1 if group k's path contains edge e
    for k, group in enumerate(paths.commodities):
        total_walk = paths.tau(k)
        walked = 0.0
        for e in group.path:
            uses[e, k] = 1.0
            walk_to[e] = min(walk_to[e], walked)
            walked += net.tau[e]
            walk_from[e] = min(walk_from[e], total_walk - walked)

    # ---- the "T must be at least ..." rules, written as rows: (row) . x <= (right side) ----
    # the unknowns are x_0 .. x_{K-1} (people in each group) and, last, T
    rows = []
    right_side = []
    for e in range(num_edges):
        if uses[e].any() and np.isfinite(net.cap[e]) and net.cap[e] > 1e-9:
            row = np.zeros(num_groups + 1)
            row[:num_groups] = uses[e] / net.cap[e]          # load_e / capacity_e
            row[num_groups] = -1.0                           # ... - T
            rows.append(row)
            right_side.append(-(delay + walk_to[e] + net.tau[e] + walk_from[e]))
    for zone in net.zones:
        row = np.zeros(num_groups + 1)
        row[num_groups] = -1.0
        quickest = min(paths.tau(k) for k in paths.by_zone[zone])
        rows.append(row)
        right_side.append(-(delay + quickest))

    # ---- every zone's people must all be assigned ----
    zone_rows = np.zeros((len(net.zones), num_groups + 1))
    zone_people = np.zeros(len(net.zones))
    for zone_number, zone in enumerate(net.zones):
        for k in paths.by_zone[zone]:
            zone_rows[zone_number, k] = 1.0
        zone_people[zone_number] = net.seats[zone_number]

    all_positive = [(0, None)] * (num_groups + 1)

    # ---- round 1: smallest T ----
    goal = np.zeros(num_groups + 1)
    goal[num_groups] = 1.0
    first = linprog(goal, A_ub=np.array(rows), b_ub=np.array(right_side), A_eq=zone_rows, b_eq=zone_people,
                    bounds=all_positive, method="highs")
    if first.status != 0:
        raise RuntimeError(first.message)
    best_T = first.x[num_groups]

    # ---- round 2: keep T (almost) that small, and walk as little as possible ----
    goal = np.zeros(num_groups + 1)
    for k in range(num_groups):
        goal[k] = paths.tau(k)
    limits = [(0, None)] * num_groups + [(0, best_T * (1 + 1e-4) + 1e-6)]
    second = linprog(goal, A_ub=np.array(rows), b_ub=np.array(right_side), A_eq=zone_rows, b_eq=zone_people,
                     bounds=limits, method="highs")
    if second.status == 0:
        people = second.x[:num_groups]
    else:
        people = first.x[:num_groups]

    # people per group -> share of the zone
    frac = np.zeros(num_groups)
    for zone_number, zone in enumerate(net.zones):
        for k in paths.by_zone[zone]:
            frac[k] = people[k] / net.seats[zone_number]
    return frac, float(best_T)


# --------------------------------------------------------------------- stage 2: which doors
def path_times_given_queues(paths, res, dt, t0_grid, weights):
    """How long each route group's path would take, given the queues seen in a simulation `res`.
    Waiting time at an edge = people in its queue / its capacity."""
    net = paths.net
    num_real_edges = len(net.edges)
    queues = res.queue_hist[:num_real_edges]
    last_step = queues.shape[1] - 1
    expected = np.zeros(len(paths.commodities))
    for k, group in enumerate(paths.commodities):
        time = t0_grid.copy()                           # start times we try
        for e in group.path:
            step = np.minimum((time / dt).astype(int), last_step)
            if net.cap[e] > 1e-9:
                wait = queues[e, step] / net.cap[e]
            else:
                wait = np.full(len(time), np.inf)       # blocked edge
            time = time + wait + max(1, round(net.tau[e] / dt)) * dt
        expected[k] = float((time * weights).sum())   # average over the start times
    return expected


def equilibrate(paths, classes, release, dt, horizon, iters=25, fixed=None, **sim_kw):
    """Let people pick the doors with the shortest queues, by trial and error.

    Repeat `iters` times:
      1. simulate with the current split
      2. for every class, find its quickest path given the queues
      3. move a small part of the class (1/(n+1)) onto that path
    `fixed` = a part of the split that never changes (e.g. people following the Crowd Managers).
    """
    net = paths.net
    num_groups = len(paths.commodities)
    if fixed is None:
        fixed = np.zeros(num_groups)

    # start: every class on its own quickest path
    class_split = []
    for zone, options, share in classes:
        split = np.zeros(num_groups)
        split[min(options, key=paths.tau)] = share
        class_split.append(split)

    # the start times people have, and how common each one is
    start_times = []
    start_weights = []
    for t in range(len(release)):
        if release[t] > 1e-6:
            start_times.append(t * dt)
            start_weights.append(release[t])
    start_times = np.array(start_times)
    start_weights = np.array(start_weights) / sum(start_weights)

    frac = fixed + sum(class_split)
    history = []
    if not classes:
        return frac, history

    for n in range(1, iters + 1):
        result = simulate(net, paths, frac, release, dt, horizon, **sim_kw)
        history.append(result.T)
        times = path_times_given_queues(paths, result, dt, start_times, start_weights)
        step_size = 1.0 / (n + 1)
        for i, (zone, options, share) in enumerate(classes):
            quickest = options[int(np.argmin(times[options]))]
            target = np.zeros(num_groups)
            target[quickest] = share
            class_split[i] = (1 - step_size) * class_split[i] + step_size * target
        frac = fixed + sum(class_split)
    return frac, history


def build_policy(paths, spec, release, dt, horizon, premovement, iters=25, exit_plan=None, **sim_kw):
    """Turn a behaviour mix, e.g. {"assigned": 0.6, "habit": 0.25, "nearest": 0.15}, into `frac`."""
    total = sum(spec.values())
    classes = []
    fixed = np.zeros(len(paths.commodities))
    for name, weight in spec.items():
        share = weight / total
        if share <= 0:
            continue
        if name == "habit":
            classes += classes_habit(paths, share)
        elif name == "nearest":
            classes += classes_nearest(paths, share)
        elif name == "ue":
            classes += classes_free(paths, share)
        elif name == "plan":
            classes += classes_fixed_exit(paths, exit_plan, share)
        elif name == "assigned":
            plan, _ = assigned(paths, premovement)
            fixed += share * plan            # Crowd Manager followers keep their assigned paths
        else:
            raise ValueError(name)
    frac, _ = equilibrate(paths, classes, release, dt, horizon, iters=iters, fixed=fixed, **sim_kw)
    return frac
