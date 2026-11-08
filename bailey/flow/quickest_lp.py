"""Model I, step 3: the fastest possible evacuation T* (a linear program).

Idea: chop time into 1-second steps and let a computer decide, for every second and every edge,
how many people walk into that edge. It has to follow these rules:

  1. An edge never takes more than (capacity x 1 s) people per step.
  2. People who enter an edge at step t arrive at its end s steps later (s = walking time in steps).
  3. Nobody appears or disappears: at every place and every step,
         people released here + people arriving + people already waiting
       = people leaving into edges + people still waiting
  4. A place cannot hold more people than its storage.

Among all plans that follow the rules, the computer picks the one where people get out as EARLY as
possible (it maximises "how early" summed over everybody). For this kind of problem the plan that
gets everybody out earliest on average also gets the LAST person out earliest, so the time when the
building is empty is T*, the best any routing could ever do.

The variables (the numbers the computer chooses):
  flow[e, t]  people entering edge e at step t
  wait[v, t]  people waiting at place v from step t to step t+1
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def transit_steps(net, dt):
    """Walking time of every edge in whole time steps (at least 1 step)."""
    steps = []
    for walk_time in net.tau:
        steps.append(max(1, int(round(walk_time / dt))))
    return np.array(steps)


class QuickestResult:
    """What solve_quickest returns."""

    def __init__(self, T, cumulative, exit_totals, edge_totals, total_person_time, status):
        self.T = T                                  # time the building is empty [s]
        self.cumulative = cumulative                # people out by the end of each step
        self.exit_totals = exit_totals              # people through each exit
        self.edge_totals = edge_totals              # people through each edge
        self.total_person_time = total_person_time  # sum of everyone's exit times [person x s]
        self.status = status                        # solver message


def solve_quickest(net, release, dt, horizon, tol=0.5):
    """Find T*. `release[t]` = share of each zone that starts moving at step t."""
    # ---- Step 1: sizes ----
    num_steps = int(np.ceil(horizon / dt))
    num_edges = len(net.edges)
    num_nodes = len(net.node_ids)
    walk_steps = transit_steps(net, dt)

    release_per_step = np.zeros(num_steps)
    for t in range(min(num_steps, len(release))):
        release_per_step[t] = release[t]

    # Every variable gets a number (its column in the big table of equations).
    def flow_var(edge, t):
        return edge * num_steps + t

    num_flow_vars = num_edges * num_steps

    def wait_var(node, t):
        return num_flow_vars + node * num_steps + t

    num_vars = num_flow_vars + num_nodes * num_steps

    # ---- Step 2: limits on each variable (rules 1 and 4) ----
    lower = np.zeros(num_vars)
    upper = [None] * num_vars                     # None = no upper limit
    for e in range(num_edges):
        for t in range(num_steps):
            if net.cap[e] <= 1e-9:
                upper[flow_var(e, t)] = 0.0                       # blocked edge
            elif t + walk_steps[e] > num_steps - 1:
                upper[flow_var(e, t)] = 0.0                       # would arrive after the end
            else:
                upper[flow_var(e, t)] = net.cap[e] * dt           # rule 1
    for v in range(num_nodes):
        for t in range(num_steps):
            if net.is_exit[v]:
                upper[wait_var(v, t)] = 0.0                       # nobody waits at an exit
            elif np.isfinite(net.storage[v]):
                upper[wait_var(v, t)] = net.storage[v]            # rule 4
    limits = list(zip(lower, upper))

    # ---- Step 3: what to optimise ----
    # Arriving at an exit at step t earns (num_steps - t) points: earlier = more points.
    # linprog minimises, so we use minus the points. A tiny cost on every movement stops the
    # computer from sending people round in circles for no reason.
    cost = np.zeros(num_vars)
    for e in range(num_edges):
        goes_to_exit = net.is_exit[net.head[e]]
        for t in range(num_steps):
            cost[flow_var(e, t)] = 1e-4
            if goes_to_exit:
                arrival_step = t + walk_steps[e]
                cost[flow_var(e, t)] -= num_steps - arrival_step

    # ---- Step 4: "nobody appears or disappears" (rule 3), one equation per place and step ----
    # equation:  (out into edges) + (still waiting) - (arriving) - (waited from before) = released here
    places = [v for v in range(num_nodes) if not net.is_exit[v]]
    equation_number = {}
    for v in places:
        for t in range(num_steps):
            equation_number[(v, t)] = len(equation_number)

    rows, cols, values = [], [], []

    def add(equation, variable, value):
        rows.append(equation)
        cols.append(variable)
        values.append(value)

    for e in range(num_edges):
        start, end = net.tail[e], net.head[e]
        for t in range(num_steps):
            if not net.is_exit[start]:
                add(equation_number[(start, t)], flow_var(e, t), 1.0)        # leaves `start` at t
            arrive = t + walk_steps[e]
            if not net.is_exit[end] and arrive < num_steps:
                add(equation_number[(end, arrive)], flow_var(e, t), -1.0)    # arrives at `end` later
    for v in places:
        for t in range(num_steps):
            add(equation_number[(v, t)], wait_var(v, t), 1.0)                 # still waiting after t
            if t > 0:
                add(equation_number[(v, t)], wait_var(v, t - 1), -1.0)        # was waiting before t

    num_equations = len(equation_number)
    A = sp.csr_matrix((values, (rows, cols)), shape=(num_equations, num_vars))
    right_side = np.zeros(num_equations)
    for zone_number, zone in enumerate(net.zones):
        v = net.nidx[zone]
        for t in range(num_steps):
            right_side[equation_number[(v, t)]] = net.seats[zone_number] * release_per_step[t]

    # ---- Step 5: let the solver find the best plan ----
    solution = linprog(cost, A_eq=A, b_eq=right_side, bounds=limits, method="highs")
    if solution.status != 0:
        raise RuntimeError(f"quickest-flow LP failed: {solution.message}")
    flow = solution.x[:num_flow_vars].reshape(num_edges, num_steps)

    # ---- Step 6: read off when people reach the exits ----
    arrivals = np.zeros(num_steps)
    exit_totals = {exit_name: 0.0 for exit_name in net.exits}
    for e in range(num_edges):
        if net.is_exit[net.head[e]]:
            exit_name = net.node_ids[net.head[e]]
            for t in range(num_steps):
                arrive = t + walk_steps[e]
                if arrive < num_steps:
                    arrivals[arrive] += flow[e, t]
                    exit_totals[exit_name] += float(flow[e, t])
    people_out = np.cumsum(arrivals)

    # the building counts as empty when fewer than `tol` people are left inside
    T = float("inf")
    for t in range(num_steps):
        if people_out[t] >= net.N - tol:
            T = float(t * dt)
            break

    total_person_time = float(sum(t * dt * arrivals[t] for t in range(num_steps)))
    return QuickestResult(T=T, cumulative=people_out, exit_totals=exit_totals, edge_totals=flow.sum(axis=1),
                          total_person_time=total_person_time, status=solution.message)
