"""Model II engine: floor-field cellular automaton with parallel update.

One person per 0.5 m cell; time step dt = cell / free level speed (~0.42 s). Each step, every moving
agent picks a neighbour cell (or stays) with probability

    p(j) ~ exp( -k_s * [F(j) - F(i)] / dt  +  k_d * D(j) ) * [j empty],

where F is the agent's floor field (time-to-go to its target exit, congestion-aware, computed on the
building as the agent BELIEVES it to be) and D is the decaying trail (herding) field. A move is then
executed with probability speed(i, j) * cell / step length (slower in seats and on stairs, diagonal
steps cover more ground). If several agents want the same cell, one random winner moves, but with
probability mu (friction) nobody does (Kirchner et al. 2003). Agents entering an exit cell leave.

Behaviour on top of the physics (bailey/behavior/choice.py): pre-movement delays, strategic exit
choice by type, Crowd Managers converting nearby agents, discovery of unknown obstructions (pause +
re-plan) and word-of-mouth spreading of that information.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import lognorm

from bailey.behavior import choice as C
from bailey.sim.fields import FieldSolver
from bailey.sim.grid import EXITS, CellGrid


@dataclass
class CAConfig:
    routing: dict[str, float]                   # behaviour mix, Model I vocabulary
    premovement: dict                           # lognormal profile from params.yaml
    blocked: set[str] = field(default_factory=set)   # aperture ids (CA) that are obstructed
    known: bool = True                          # obstruction known at t = 0
    crowd_managers: bool = False
    guided_shares: dict[str, dict[str, float]] | None = None          # plan as people are told it
    guided_shares_informed: dict[str, dict[str, float]] | None = None # plan once the obstruction is known
    exit_capacity: dict[str, float] | None = None                     # p/s, for rational agents
    snapshot_times: tuple[float, ...] = ()


@dataclass
class CAResult:
    T: float                     # time the last person left (inf if not everybody did)
    exit_time: np.ndarray        # per agent (nan if still inside)
    exit_id: np.ndarray          # per agent index into EXITS (-1 if still inside)
    agent_type: np.ndarray
    agent_zone: np.ndarray
    start_time: np.ndarray
    dt: float
    snapshots: dict[float, tuple[np.ndarray, np.ndarray]]   # t -> (cells, types) of agents inside
    occupancy_time: np.ndarray   # seconds each cell spent occupied (congestion heat map)
    remaining: int

    def curve(self, t_max: float, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(0, t_max + step, step)
        et = np.sort(self.exit_time[np.isfinite(self.exit_time)])
        return t, len(self.exit_time) - np.searchsorted(et, t, side="right")

    def percentile(self, q: float) -> float:
        et = self.exit_time
        return float(np.nanpercentile(np.where(np.isfinite(et), et, np.inf), q))


class Engine:
    """Static pieces shared by every run on the same grid and obstruction set."""

    def __init__(self, grid: CellGrid, params: dict, blocked: set[str] = frozenset()):
        self.grid, self.P = grid, params
        ca = params["ca"]
        self.v_free = params["hydraulic"]["free_speed_factor"] * params["hydraulic"]["k"]["level"]
        self.dt = ca["cell"] / self.v_free
        self.blocked = set(blocked)
        self.fs_actual = FieldSolver(grid, self.v_free, self.blocked)
        self.fs_plan = FieldSolver(grid, self.v_free) if self.blocked else self.fs_actual
        self.F_static = {"actual": self.fs_actual.solve(), "plan": self.fs_plan.solve()}
        # movement neighbour table on the ACTUAL building (people cannot walk through obstructions)
        E, L = grid.edges(self.blocked)
        src = np.concatenate([E[:, 0], E[:, 1]]); dst = np.concatenate([E[:, 1], E[:, 0]])
        ln = np.concatenate([L, L])
        order = np.argsort(src, kind="stable")
        src, dst, ln = src[order], dst[order], ln[order]
        deg = np.bincount(src, minlength=grid.n)
        D = int(deg.max())
        self.neigh = -np.ones((grid.n, D), int)
        self.nlen = np.ones((grid.n, D))
        start = np.concatenate([[0], np.cumsum(deg)[:-1]])
        slot = np.arange(len(src)) - start[src]
        self.neigh[src, slot] = dst
        self.nlen[src, slot] = ln
        self.exit_of_cell = grid.exit_index()
        self.zone_names = list(grid.zone_cells)
        # obstruction centres (for discovery)
        self.block_centres = [grid.apertures[a].center for a in self.blocked if a in grid.apertures]
        # exits that are physically unreachable once the obstruction is known
        self.reach_actual = np.isfinite(self.F_static["actual"]).any(axis=1)

    # ---------------------------------------------------------------------------------- run
    def run(self, cfg: CAConfig, seed: int, seats: dict[str, int]) -> CAResult:
        P, ca, g = self.P, self.P["ca"], self.grid
        rng = np.random.default_rng(seed)
        dt = self.dt
        n_ex = len(EXITS)

        # ---------------- agents
        pos, zone = [], []
        for zi, z in enumerate(self.zone_names):
            cells = rng.choice(g.zone_cells[z], size=seats[z], replace=False)
            pos.append(cells); zone.append(np.full(seats[z], zi))
        pos = np.concatenate(pos); zone = np.concatenate(zone)
        A = len(pos)
        mix = {k: v for k, v in cfg.routing.items() if v > 0}
        names = list(mix); probs = np.array([mix[k] for k in names]) / sum(mix.values())
        atype = np.array([C.TYPE_OF[names[i]] for i in rng.choice(len(names), size=A, p=probs)])
        k_d = np.where(rng.random(A) < ca["herding_share"], ca["k_d_herd"], ca["k_d"])
        pm = cfg.premovement
        dist = lognorm(s=pm["sigma"], scale=pm["median"])
        lo, hi = dist.cdf(pm["t_min"]), pm["q_cap"]
        start = dist.ppf(lo + rng.random(A) * (hi - lo))

        has_block = bool(self.blocked)
        informed = np.full(A, (not has_block) or cfg.known)
        cm_informed = (not has_block) or cfg.known
        alive = np.ones(A, bool)
        chosen = np.zeros(A, bool)
        target = np.zeros(A, int)
        wait_until = np.zeros(A)
        last_reeval = np.zeros(A)
        exit_time = np.full(A, np.nan); exit_id = -np.ones(A, int)

        occ = np.zeros(g.n, bool); occ[pos] = True
        dyn = np.zeros(g.n)
        occ_time = np.zeros(g.n)
        all_open = np.ones(n_ex, bool)
        open_actual = self.reach_actual.copy()

        def shares_table(sh):
            tab = np.zeros((len(self.zone_names), n_ex))
            if sh:
                for zi, z in enumerate(self.zone_names):
                    for e, v in sh.get(z, {}).items():
                        tab[zi, EXITS.index(e)] = v
            return tab
        sh_plan = shares_table(cfg.guided_shares)
        sh_inf = shares_table(cfg.guided_shares_informed or cfg.guided_shares)
        caps = np.array([(cfg.exit_capacity or {}).get(e, 2.0) for e in EXITS])

        cm_cells = []
        if cfg.crowd_managers:
            from bailey.geometry.schematic import polar
            for layer, r, th in ca["cm_positions"]:
                cm_cells.append(g.nearest_cell(layer, *polar(r, th)))
        cm_cells = np.array(cm_cells, int)
        p_cm = 1 - np.exp(-ca["cm_switch_rate"] * dt)

        F_dyn = {"actual": self.F_static["actual"], "plan": self.F_static["plan"]}
        next_refresh = 0.0
        snapshots = {}
        snap_left = sorted(cfg.snapshot_times)
        k_s, mu = ca["k_s"], ca["friction"]
        horizon = ca["horizon"]
        t = 0.0
        step = 0

        def choose(idx):
            """(Re)choose exits for agents idx according to their type and knowledge."""
            if len(idx) == 0:
                return
            inf_ = informed[idx]
            for flag in (True, False):
                sel = idx[inf_ == flag]
                if len(sel) == 0:
                    continue
                key = "actual" if flag else "plan"
                allowed = open_actual if flag else all_open
                Fs = self.F_static[key]
                cells = pos[sel]
                near = C.nearest_exit(Fs, cells, allowed)
                ty = atype[sel]
                tgt = near.copy()
                m = ty == C.HABIT
                tgt[m] = C.habit_exit(Fs, cells[m], allowed)
                m = ty == C.GUIDED
                if m.any():
                    tab = sh_inf if (flag and has_block) else sh_plan
                    tgt[m] = C.guided_exit(rng, tab, zone[sel][m], allowed, near[m])
                m = ty == C.RATIONAL
                if m.any():
                    Fd = F_dyn[key]
                    ahead = C.people_ahead(Fd, pos, target, alive & chosen, cells[m])
                    tgt[m] = np.argmin(C.rational_cost(Fd, cells[m], ahead, caps, allowed), axis=0)
                target[sel] = tgt
            chosen[idx] = True

        while alive.any() and t <= horizon:
            # ---------------- snapshots
            while snap_left and t >= snap_left[0]:
                snapshots[snap_left.pop(0)] = (pos[alive].copy(), atype[alive].copy())
            # ---------------- congestion-aware fields
            if t >= next_refresh:
                beta = ca["congestion_beta"]
                F_dyn["actual"] = self.fs_actual.solve(occ, beta)
                F_dyn["plan"] = self.fs_plan.solve(occ, beta) if (has_block and not informed.all()) else F_dyn["actual"]
                next_refresh = t + ca["field_refresh"]

            active = alive & (t >= start)
            # newly moving agents choose an exit
            choose(np.where(active & ~chosen)[0])
            # rational agents reconsider periodically (with hysteresis)
            re = np.where(active & (atype == C.RATIONAL) & (t - last_reeval >= ca["rational_reeval"]))[0]
            if len(re):
                last_reeval[re] = t
                for flag in (True, False):
                    sel = re[informed[re] == flag]
                    if len(sel) == 0:
                        continue
                    key = "actual" if flag else "plan"
                    allowed = open_actual if flag else all_open
                    ahead = C.people_ahead(F_dyn[key], pos, target, active & chosen, pos[sel])
                    cost = C.rational_cost(F_dyn[key], pos[sel], ahead, caps, allowed)
                    best = np.argmin(cost, axis=0)
                    cur = cost[target[sel], np.arange(len(sel))]
                    sw = cost[best, np.arange(len(sel))] < cur - ca["rational_switch_gain"]
                    target[sel[sw]] = best[sw]

            # ---------------- Crowd Managers
            if len(cm_cells):
                cm_knows = cm_informed or (t >= ca["cm_radio_delay"])
                cand = np.where(active)[0]
                if len(cand):
                    d = np.full(len(cand), np.inf)
                    for c in cm_cells:
                        same = g.layer[pos[cand]] == g.layer[c]
                        dd = np.hypot(g.x[pos[cand]] - g.x[c], g.y[pos[cand]] - g.y[c])
                        d = np.where(same, np.minimum(d, dd), d)
                    near_cm = cand[np.nan_to_num(d, nan=np.inf) <= ca["cm_radius"]]
                    if cm_knows and has_block:
                        newly = near_cm[~informed[near_cm]]
                        informed[newly] = True
                        choose(newly)
                    conv = near_cm[(atype[near_cm] != C.GUIDED) & (rng.random(len(near_cm)) < p_cm)]
                    if len(conv):
                        atype[conv] = C.GUIDED
                        choose(conv)

            # ---------------- discovery of an unknown obstruction + word of mouth
            if has_block and not informed.all():
                un = np.where(active & ~informed)[0]
                if len(un):
                    found = np.zeros(len(un), bool)
                    for layer, cx, cy in self.block_centres:
                        same = g.layer[pos[un]] == layer
                        dd = np.hypot(g.x[pos[un]] - cx, g.y[pos[un]] - cy)
                        found |= same & (np.nan_to_num(dd, nan=np.inf) <= ca["see_radius"])
                    disc = un[found]
                    informed[disc] = True
                    wait_until[disc] = t + P["blocked_discovery_delay"]
                    choose(disc)
                    rest = un[~found]
                    if len(rest):
                        inf_cell = np.zeros(g.n, bool); inf_cell[pos[alive & informed]] = True
                        nb = self.neigh[pos[rest]]
                        heard = (inf_cell[np.maximum(nb, 0)] & (nb >= 0)).any(1)
                        learn = rest[heard & (rng.random(len(rest)) < ca["info_spread"])]
                        informed[learn] = True
                        choose(learn)

            # ---------------- movement
            mv = np.where(active & (t >= wait_until))[0]
            if len(mv):
                key_idx = np.where(informed[mv], 0, 1)
                Fstack = np.stack([F_dyn["actual"], F_dyn["plan"]])
                cells = pos[mv]
                nb = self.neigh[cells]
                valid = nb >= 0
                nbc = np.maximum(nb, 0)
                valid &= ~occ[nbc]
                f0 = Fstack[key_idx, target[mv], cells]
                fn = Fstack[key_idx[:, None], target[mv][:, None], nbc]
                with np.errstate(invalid="ignore", over="ignore"):
                    logit = -k_s * (fn - f0[:, None]) / dt + k_d[mv, None] * dyn[nbc]
                logit = np.where(valid & np.isfinite(logit), logit, -np.inf)
                stay = k_d[mv] * dyn[cells]
                full = np.column_stack([stay, logit])
                full -= full.max(1, keepdims=True)
                w = np.exp(full)
                w /= w.sum(1, keepdims=True)
                pick = (rng.random(len(mv))[:, None] > np.cumsum(w, 1)).sum(1)
                pick = np.minimum(pick, w.shape[1] - 1)
                moving = pick > 0
                sl = pick - 1
                dest = np.where(moving, nb[np.arange(len(mv)), np.maximum(sl, 0)], -1)
                steplen = self.nlen[cells, np.maximum(sl, 0)]
                sp = 0.5 * (g.speed[cells] + g.speed[np.maximum(dest, 0)])
                accept = moving & (rng.random(len(mv)) < np.minimum(1.0, sp * g.h / steplen))
                who = mv[accept]; dest = dest[accept]
                if len(who):
                    # conflicts: one random winner per cell, none with probability mu
                    order = rng.permutation(len(who))
                    who, dest = who[order], dest[order]
                    first = np.unique(dest, return_index=True, return_counts=True)
                    _, idx_first, counts = first
                    keep = np.zeros(len(who), bool)
                    contested = counts > 1
                    keep[idx_first[~contested]] = True
                    ok = idx_first[contested][rng.random(contested.sum()) >= mu]
                    keep[ok] = True
                    who, dest = who[keep], dest[keep]
                    old = pos[who]
                    occ[old] = False
                    dyn[old] += 1.0
                    pos[who] = dest
                    ex = self.exit_of_cell[dest]
                    out = ex >= 0
                    alive[who[out]] = False
                    exit_time[who[out]] = t + dt
                    exit_id[who[out]] = ex[out]
                    occ[dest[~out]] = True
                # ---- counter-flow exchange: two blocked agents facing each other swap cells
                moved = np.zeros(A, bool); moved[who] = True
                stuck = mv[~moved[mv] & alive[mv]]
                if len(stuck) > 1 and ca["swap_prob"] > 0:
                    sc = pos[stuck]
                    nb_s = self.neigh[sc]
                    fs = Fstack[np.where(informed[stuck], 0, 1)[:, None], target[stuck][:, None], np.maximum(nb_s, 0)]
                    fs = np.where(nb_s >= 0, fs, np.inf)
                    best = nb_s[np.arange(len(stuck)), np.argmin(fs, 1)]
                    agent_at = -np.ones(g.n, int); agent_at[sc] = np.arange(len(stuck))
                    j = agent_at[np.maximum(best, 0)]
                    pair = (best >= 0) & (j >= 0)
                    i_idx = np.where(pair)[0]
                    j_idx = j[pair]
                    mutual = (best[j_idx] == sc[i_idx]) & (i_idx < j_idx)
                    i_idx, j_idx = i_idx[mutual], j_idx[mutual]
                    go = rng.random(len(i_idx)) < ca["swap_prob"]
                    a_, b_ = stuck[i_idx[go]], stuck[j_idx[go]]
                    pos[a_], pos[b_] = pos[b_].copy(), pos[a_].copy()
            dyn *= (1.0 - ca["dyn_decay"])
            occ_time += occ * dt
            t += dt
            step += 1

        T = float(np.nanmax(exit_time)) if not alive.any() else float("inf")
        return CAResult(T=T, exit_time=exit_time, exit_id=exit_id, agent_type=atype, agent_zone=zone,
                        start_time=start, dt=dt, snapshots=snapshots, occupancy_time=occ_time,
                        remaining=int(alive.sum()))
