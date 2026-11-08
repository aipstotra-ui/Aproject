"""Model I, step 6: run one whole scenario.

For a scenario (e.g. scenarios/concert_cm.yaml) this:
  1. builds the network (with a blocked door, if asked)
  2. lists the route options (PathSet)
  3. for the scenario's behaviour mix, and any extra mixes we want to compare, works out how people
     split over the routes and simulates the evacuation
  4. optionally computes the quick lower limits (bounds.py) and the fastest possible time T*
"""
from pathlib import Path

import yaml

from bailey.flow.bounds import lower_bounds
from bailey.flow.fluid import PathSet, simulate
from bailey.flow.network import make_network
from bailey.flow.policies import build_policy
from bailey.flow.quickest_lp import solve_quickest
from bailey.params import ROOT, load_params, premovement_release

SCENARIO_DIR = ROOT / "scenarios"

# the "pure" behaviours we compare every scenario with
REFERENCE_POLICIES = {
    "habit": {"habit": 1.0},
    "nearest": {"nearest": 1.0},
    "ue": {"ue": 1.0},
    "assigned": {"assigned": 1.0},
}


def load_scenario(name_or_path):
    """Read a scenario file, by name ("concert_cm") or by path."""
    path = Path(name_or_path)
    if not path.exists():
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class ScenarioRun:
    """Everything one scenario run produced."""

    def __init__(self, scenario, net, blocked, known):
        self.scenario = scenario
        self.net = net
        self.blocked = blocked
        self.known = known
        self.bounds = None        # lower limits (dict), if computed
        self.quickest = None      # T* result, if computed
        self.sims = {}            # behaviour name -> FluidResult


def run_scenario(scn, blocked=None, known=True, policies=None, with_lp=True, with_bounds=True, params=None):
    """Evaluate one scenario.

    blocked   {edge id: capacity share left}
    known     False = nobody knows about the blocked door until they reach it (even the Crowd
              Managers plan on the intact building)
    policies  extra behaviour mixes to run, {label: mix}; the scenario's own mix is called "scenario"
    """
    if params is None:
        params = load_params()
    if blocked is None:
        blocked = {}
    dt = params["numerics"]["dt"]
    horizon = params["numerics"]["horizon"]
    tol = params["numerics"]["done_tolerance"]
    reaction = params["premovement"][scn["premovement"]]
    release = premovement_release(params, scn["premovement"], dt, horizon)

    # ---- 1 + 2: network and route options ----
    net = make_network(scn["mode"], blocked, known=known, params=params)
    paths = PathSet.build(net)

    # People plan on the building as they BELIEVE it to be. If the blocked door is unknown, they plan
    # on the intact building (same route options, but the blocked door looks open).
    if known:
        planning_net = net
    else:
        planning_net = make_network(scn["mode"], {}, params=params)
    planning_paths = PathSet(planning_net, paths.commodities, paths.by_zone)

    settings = {"tol": tol, "discovery_delay": params["blocked_discovery_delay"]}

    # ---- 3: simulate every behaviour mix ----
    run = ScenarioRun(scn, net, blocked, known)
    mixes = {"scenario": scn["routing"]}
    if policies:
        mixes.update(policies)
    for label, mix in mixes.items():
        frac = build_policy(planning_paths, mix, release, dt, horizon, reaction, **settings)
        run.sims[label] = simulate(net, paths, frac, release, dt, horizon, **settings)

    # ---- 4: lower limits and fastest possible time ----
    if with_bounds:
        run.bounds = lower_bounds(make_network(scn["mode"], blocked, params=params), scn["premovement"])
    if with_lp:
        # T* can never be later than our quickest simulation, so we only need to look that far ahead
        finished = [result.T for result in run.sims.values() if result.T != float("inf")]
        if finished:
            lp_horizon = min(finished) + 10 * dt
        else:
            lp_horizon = horizon
        full_net = make_network(scn["mode"], blocked, params=params)
        run.quickest = solve_quickest(full_net, release, dt, lp_horizon, tol=tol)
    return run


def summary_rows(run):
    """One table row per behaviour (and one for T*), for the results CSV."""
    scn = run.scenario
    if run.blocked:
        blocked_text = ";".join(f"{name}x{share:.2f}" for name, share in run.blocked.items())
    else:
        blocked_text = "none"
    common = {"scenario": scn["name"], "question": scn["question"], "blocked": blocked_text, "known": run.known}

    rows = []
    if run.quickest is not None:
        row = dict(common)
        row["policy"] = "T*"
        row["T"] = run.quickest.T
        row["mean_exit"] = run.quickest.total_person_time / run.net.N
        for exit_name, people in run.quickest.exit_totals.items():
            row["out_" + exit_name] = round(people, 1)
        rows.append(row)
    for label, result in run.sims.items():
        row = dict(common)
        row["policy"] = label
        row["T"] = result.T
        row["mean_exit"] = result.mean_exit_time
        for exit_name, people in result.exit_totals.items():
            row["out_" + exit_name] = round(people, 1)
        rows.append(row)
    if run.bounds is not None:
        for row in rows:
            row["T_lower_bound"] = run.bounds["T_lower_bound"]
    return rows
