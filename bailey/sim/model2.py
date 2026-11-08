"""Model II, step 6: run one scenario many times (Monte Carlo) and summarise.

Because people's choices and steps are random, every run comes out a little different, so we run each
scenario 30 times (seeds 0, 1, 2, ...) and report the average, the spread and the chance of finishing
within 180 s.

Two things come from Model I:
  * the Crowd Managers' plan: Model I's best split of every seating zone over the exits
  * the exit capacities that rational people use to estimate waiting times
Blocked doors are named like Model I edges. For the five front doors (one Model I edge with 5 leaves)
we close the middle one.
"""
import networkx as nx
import numpy as np

from bailey.flow.fluid import PathSet
from bailey.flow.model1 import load_scenario
from bailey.flow.network import load_building, make_network
from bailey.flow.policies import assigned
from bailey.params import load_params
from bailey.sim.engine import CAConfig, Engine
from bailey.sim.grid import EXITS, build_grid


def ca_blocked(grid, blocked):
    """Turn a Model I blockage {edge name: capacity share left} into the names of the grid's doors to close."""
    closed = set()
    for edge_name, share_left in blocked.items():
        # the grid's doors for this edge, e.g. "FOYER->OUT_FRONT#0" ... "#4" (or just one)
        leaves = sorted(name for name in grid.apertures if name.split("#")[0] == edge_name)
        if not leaves:
            raise KeyError(f"no CA aperture for Model I edge {edge_name}")
        if len(leaves) > 1:
            how_many = int(round((1 - share_left) * len(leaves)))
        else:
            how_many = 1 if share_left < 1 else 0
        # close the middle leaves first
        middle = len(leaves) // 2
        by_distance_from_middle = sorted(leaves, key=lambda name: abs(leaves.index(name) - middle))
        for name in by_distance_from_middle[:how_many]:
            closed.add(name)
    return closed


def guided_shares(mode, blocked, premovement):
    """The Crowd Managers' plan from Model I: {zone: {exit: share of the zone}}."""
    net = make_network(mode, blocked)
    paths = PathSet.build(net)
    frac, _ = assigned(paths, premovement)
    plan = {zone: {} for zone in net.zones}
    for k, group in enumerate(paths.commodities):
        if frac[k] > 1e-9:
            plan[group.zone][group.exit] = plan[group.zone].get(group.exit, 0.0) + float(frac[k])
    return plan


def exit_capacities(mode, blocked):
    """For each exit: the most people per second that could reach it from all seats (Model I max-flow)."""
    net = make_network(mode, blocked)
    G = nx.DiGraph()
    for i, edge in enumerate(net.edges):
        if net.cap[i] > 1e-9:
            G.add_edge(edge["from"], edge["to"], capacity=float(net.cap[i]))
    for zone in net.zones:
        G.add_edge("__SRC__", zone)
    capacities = {}
    for exit_name in EXITS:
        if G.has_node(exit_name):
            capacities[exit_name] = max(float(nx.maximum_flow_value(G, "__SRC__", exit_name)), 1e-6)
        else:
            capacities[exit_name] = 1e-6
    return capacities


class MCResult:
    """All runs of one scenario."""

    def __init__(self, scenario, blocked, known, variant):
        self.scenario = scenario
        self.blocked = blocked
        self.known = known
        self.variant = variant
        self.runs = []

    @property
    def T(self):
        """Evacuation time of every run."""
        return np.array([run.T for run in self.runs])

    def summary(self):
        """One row for the results CSV."""
        times = self.T
        finished = times[np.isfinite(times)]

        # average number of people per exit
        usage = []
        for exit_number in range(len(EXITS)):
            counts = [np.sum(run.exit_id == exit_number) for run in self.runs]
            usage.append(np.mean(counts))

        if self.blocked:
            blocked_text = ";".join(f"{name}x{share:.2f}" for name, share in self.blocked.items())
        else:
            blocked_text = "none"

        row = {
            "scenario": self.scenario["name"],
            "question": self.scenario["question"],
            "variant": self.variant,
            "blocked": blocked_text,
            "known": self.known,
            "runs": len(times),
            "T_mean": float(finished.mean()) if len(finished) else np.inf,
            "T_sd": float(finished.std(ddof=1)) if len(finished) > 1 else 0.0,
            "T_p05": float(np.percentile(times, 5)),
            "T_p95": float(np.percentile(times, 95)),
            "P_T_le_180": float(np.mean(times <= 180.0)),
            "T95pct_people": float(np.mean([run.percentile(95) for run in self.runs])),
            "mean_exit": float(np.mean([np.nanmean(run.exit_time) for run in self.runs])),
            "stuck_runs": int(np.sum(~np.isfinite(times))),
        }
        for exit_name, people in zip(EXITS, usage):
            row["out_" + exit_name] = round(float(people), 1)
        return row


class Model2:
    def __init__(self, params=None):
        if params is None:
            params = load_params()
        self.P = params
        self.grid = build_grid(params)
        self.seats = {}
        for node in load_building()["nodes"]:
            if node["kind"] == "zone":
                self.seats[node["id"]] = node["seats"]
        self._engines = {}                # one Engine per set of blocked doors (re-used between runs)

    def engine(self, blocked_doors):
        key = frozenset(blocked_doors)
        if key not in self._engines:
            self._engines[key] = Engine(self.grid, self.P, blocked_doors)
        return self._engines[key]

    def run(self, scn, blocked=None, known=True, runs=None, routing=None, crowd_managers=None,
            variant="full", seed0=0, snapshot_times=()):
        """Run scenario `scn` (name or dict) `runs` times.

        blocked         {Model I edge name: capacity share left}
        known           False = the blockage is a surprise
        routing         use this behaviour mix instead of the scenario's own
        crowd_managers  True/False to override the scenario's setting
        """
        if isinstance(scn, str):
            scn = load_scenario(scn)
        if blocked is None:
            blocked = {}
        reaction = self.P["premovement"][scn["premovement"]]
        if routing is None:
            routing = scn["routing"]
        if crowd_managers is None:
            crowd_managers = scn.get("crowd_managers", False)

        # the Crowd Managers' plan is only needed if someone follows it
        needs_plan = crowd_managers or ("assigned" in routing) or ("plan" in routing)
        plan_after = None
        plan_before = None
        if needs_plan:
            plan_after = guided_shares(scn["mode"], blocked, reaction)        # knowing about the blockage
            if blocked and not known:
                plan_before = guided_shares(scn["mode"], {}, reaction)       # before they find out
            else:
                plan_before = plan_after

        capacities = exit_capacities(scn["mode"], blocked if known else {})
        settings = CAConfig(routing=routing, premovement=reaction, blocked=ca_blocked(self.grid, blocked),
                            known=known, crowd_managers=crowd_managers, guided_shares=plan_before,
                            guided_shares_informed=plan_after, exit_capacity=capacities,
                            snapshot_times=tuple(snapshot_times))
        engine = self.engine(settings.blocked)

        result = MCResult(scn, blocked, known, variant)
        if runs is None:
            runs = self.P["ca"]["runs"]
        for number in range(runs):
            result.runs.append(engine.run(settings, seed=seed0 + number, seats=self.seats))
        return result
