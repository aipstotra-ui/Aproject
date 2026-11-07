"""Model II driver: scenario file -> Monte Carlo runs of the cellular automaton.

Links to Model I:
  * Crowd Managers (and 'assigned' agents) tell people the exit from Model I's system-optimal LP
    assignment for their zone (bailey.flow.policies.assigned), recomputed when an obstruction is known.
  * Rational agents estimate exit capacity as Model I's max flow from all seats to that exit.
  * Obstructions use Model I edge ids. A multi-leaf door (the 5 foyer doors) loses its middle leaf.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from bailey.flow.fluid import PathSet
from bailey.flow.model1 import load_scenario
from bailey.flow.network import load_building, make_network
from bailey.flow.policies import assigned
from bailey.params import load_params
from bailey.sim.engine import CAConfig, CAResult, Engine
from bailey.sim.grid import EXITS, CellGrid, build_grid


def ca_blocked(grid: CellGrid, blocked: dict[str, float]) -> set[str]:
    """Model I obstruction {edge id: remaining fraction} -> set of CA aperture ids."""
    out = set()
    for eid, frac in blocked.items():
        leaves = sorted(a for a in grid.apertures if a.split("#")[0] == eid)
        if not leaves:
            raise KeyError(f"no CA aperture for Model I edge {eid}")
        n_block = int(round((1 - frac) * len(leaves))) if len(leaves) > 1 else (1 if frac < 1 else 0)
        mid = len(leaves) // 2
        order = sorted(leaves, key=lambda a: abs(leaves.index(a) - mid))
        out |= set(order[:n_block])
    return out


def guided_shares(mode: str, blocked: dict[str, float], premovement: dict) -> dict[str, dict[str, float]]:
    """Zone -> {exit: share} from Model I's static system-optimal assignment."""
    net = make_network(mode, blocked)
    ps = PathSet.build(net)
    frac, _ = assigned(ps, premovement)
    out: dict[str, dict[str, float]] = {z: {} for z in net.zones}
    for k, c in enumerate(ps.commodities):
        if frac[k] > 1e-9:
            out[c.zone][c.exit] = out[c.zone].get(c.exit, 0.0) + float(frac[k])
    return out


def exit_capacities(mode: str, blocked: dict[str, float]) -> dict[str, float]:
    """Max flow [p/s] from all seats to each single exit (Model I network)."""
    net = make_network(mode, blocked)
    G = nx.DiGraph()
    for i, e in enumerate(net.edges):
        if net.cap[i] > 1e-9:
            G.add_edge(e["from"], e["to"], capacity=float(net.cap[i]))
    for z in net.zones:
        G.add_edge("__SRC__", z)
    caps = {}
    for x in EXITS:
        caps[x] = float(nx.maximum_flow_value(G, "__SRC__", x)) if G.has_node(x) else 1e-6
    return {x: max(c, 1e-6) for x, c in caps.items()}


@dataclass
class MCResult:
    scenario: dict
    blocked: dict[str, float]
    known: bool
    variant: str
    runs: list[CAResult] = field(default_factory=list)

    @property
    def T(self) -> np.ndarray:
        return np.array([r.T for r in self.runs])

    def summary(self) -> dict:
        T = self.T
        fin = T[np.isfinite(T)]
        usage = np.mean([[np.sum(r.exit_id == i) for i in range(len(EXITS))] for r in self.runs], axis=0)
        return {
            "scenario": self.scenario["name"], "question": self.scenario["question"], "variant": self.variant,
            "blocked": ";".join(f"{k}x{v:.2f}" for k, v in self.blocked.items()) or "none", "known": self.known,
            "runs": len(T), "T_mean": float(fin.mean()) if len(fin) else np.inf,
            "T_sd": float(fin.std(ddof=1)) if len(fin) > 1 else 0.0,
            "T_p05": float(np.percentile(T, 5)), "T_p95": float(np.percentile(T, 95)),
            "P_T_le_180": float(np.mean(T <= 180.0)),
            "T95pct_people": float(np.mean([r.percentile(95) for r in self.runs])),
            "mean_exit": float(np.mean([np.nanmean(r.exit_time) for r in self.runs])),
            "stuck_runs": int(np.sum(~np.isfinite(T))),
            **{f"out_{x}": round(float(u), 1) for x, u in zip(EXITS, usage)},
        }


class Model2:
    def __init__(self, params: dict | None = None):
        self.P = params or load_params()
        self.grid = build_grid(self.P)
        self.seats = {n["id"]: n["seats"] for n in load_building()["nodes"] if n["kind"] == "zone"}
        self._engines: dict[frozenset, Engine] = {}

    def engine(self, blocked_ca: set[str]) -> Engine:
        key = frozenset(blocked_ca)
        if key not in self._engines:
            self._engines[key] = Engine(self.grid, self.P, blocked_ca)
        return self._engines[key]

    def run(self, scn: dict | str, *, blocked: dict[str, float] | None = None, known: bool = True,
            runs: int | None = None, routing: dict | None = None, crowd_managers: bool | None = None,
            variant: str = "full", seed0: int = 0, snapshot_times=()) -> MCResult:
        scn = load_scenario(scn) if isinstance(scn, str) else scn
        blocked = dict(blocked or {})
        pm = self.P["premovement"][scn["premovement"]]
        routing = routing or scn["routing"]
        cms = scn.get("crowd_managers", False) if crowd_managers is None else crowd_managers
        need_plan = cms or any(k in ("assigned", "plan") for k in routing)
        sh_inf = guided_shares(scn["mode"], blocked, pm) if need_plan else None
        sh_plan = (guided_shares(scn["mode"], {}, pm) if (need_plan and blocked and not known) else sh_inf)
        cfg = CAConfig(routing=routing, premovement=pm, blocked=ca_blocked(self.grid, blocked), known=known,
                       crowd_managers=cms, guided_shares=sh_plan, guided_shares_informed=sh_inf,
                       exit_capacity=exit_capacities(scn["mode"], blocked if known else {}),
                       snapshot_times=tuple(snapshot_times))
        eng = self.engine(cfg.blocked)
        out = MCResult(scn, blocked, known, variant)
        for s in range(runs or self.P["ca"]["runs"]):
            out.runs.append(eng.run(cfg, seed=seed0 + s, seats=self.seats))
        return out
