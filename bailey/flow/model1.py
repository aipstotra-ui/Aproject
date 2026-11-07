"""Model I driver: scenario file -> bounds, quickest-flow optimum T*, and fluid simulations per policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from bailey.flow.bounds import lower_bounds
from bailey.flow.fluid import FluidResult, PathSet, simulate
from bailey.flow.network import Network, make_network
from bailey.flow.policies import build_policy
from bailey.flow.quickest_lp import QuickestResult, solve_quickest
from bailey.params import ROOT, load_params, premovement_release

SCENARIO_DIR = ROOT / "scenarios"
REFERENCE_POLICIES = {"habit": {"habit": 1.0}, "nearest": {"nearest": 1.0},
                      "ue": {"ue": 1.0}, "assigned": {"assigned": 1.0}}


def load_scenario(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = SCENARIO_DIR / f"{name_or_path}.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


@dataclass
class ScenarioRun:
    scenario: dict
    net: Network
    blocked: dict[str, float]
    known: bool
    bounds: dict | None = None
    quickest: QuickestResult | None = None
    sims: dict[str, FluidResult] = field(default_factory=dict)


def run_scenario(scn: dict, *, blocked: dict[str, float] | None = None, known: bool = True,
                 policies: dict[str, dict] | None = None, with_lp: bool = True, with_bounds: bool = True,
                 params: dict | None = None) -> ScenarioRun:
    """Evaluate one scenario. `policies` maps a label to a routing mix; the scenario's own mix is
    always run under the label 'scenario'.

    Unknown obstruction (known=False): routes - including the Crowd Managers' LP assignment - are
    planned on the intact building; people discover the obstruction on arrival and re-route.
    """
    P = params or load_params()
    dt, H = P["numerics"]["dt"], P["numerics"]["horizon"]
    tol = P["numerics"]["done_tolerance"]
    pm = P["premovement"][scn["premovement"]]
    rel = premovement_release(P, scn["premovement"], dt, H)
    blocked = dict(blocked or {})

    net = make_network(scn["mode"], blocked, known=known, params=P)
    paths = PathSet.build(net)
    planning_net = net if known else make_network(scn["mode"], {}, params=P)
    planning_paths = PathSet(planning_net, paths.commodities, paths.by_zone)
    sim_kw = dict(tol=tol, discovery_delay=P["blocked_discovery_delay"])

    run = ScenarioRun(scn, net, blocked, known)
    mixes = {"scenario": scn["routing"], **(policies or {})}
    for label, spec in mixes.items():
        frac = build_policy(planning_paths, spec, rel, dt, H, pm, **sim_kw)
        run.sims[label] = simulate(net, paths, frac, rel, dt, H, **sim_kw)
    if with_bounds:
        run.bounds = lower_bounds(make_network(scn["mode"], blocked, params=P), scn["premovement"])
    if with_lp:
        finite = [r.T for r in run.sims.values() if np.isfinite(r.T)]
        horizon = (min(finite) + 10 * dt) if finite else H   # any fluid run is LP-feasible -> T* <= min T
        run.quickest = solve_quickest(make_network(scn["mode"], blocked, params=P), rel, dt, horizon, tol=tol)
    return run


def summary_rows(run: ScenarioRun) -> list[dict]:
    scn = run.scenario
    base = {"scenario": scn["name"], "question": scn["question"],
            "blocked": ";".join(f"{k}x{v:.2f}" for k, v in run.blocked.items()) or "none",
            "known": run.known}
    rows = []
    if run.quickest is not None:
        rows.append({**base, "policy": "T*", "T": run.quickest.T,
                     "mean_exit": run.quickest.total_person_time / run.net.N,
                     **{f"out_{k}": round(v, 1) for k, v in run.quickest.exit_totals.items()}})
    for label, r in run.sims.items():
        rows.append({**base, "policy": label, "T": r.T, "mean_exit": r.mean_exit_time,
                     **{f"out_{k}": round(v, 1) for k, v in r.exit_totals.items()}})
    if run.bounds is not None:
        for row in rows:
            row["T_lower_bound"] = run.bounds["T_lower_bound"]
    return rows
