"""Verification of Model I against analytic cases and internal consistency."""
import math

import numpy as np
import pytest

from bailey.flow.bounds import lower_bounds
from bailey.flow.fluid import PathSet, simulate
from bailey.flow.network import Network, load_building, make_network, one_leaf_blocked
from bailey.flow.policies import build_policy
from bailey.flow.quickest_lp import solve_quickest
from bailey.params import capacity, free_speed, load_params, premovement_release

P = load_params()


def toy_room(n_people=200, door_width=1.2, length=10.0, area=None):
    """One zone -> one door -> exit."""
    return {
        "nodes": [
            {"id": "Z", "x": 0, "y": 0, "z": 0, "kind": "zone", "area": None, "seats": n_people, "label": ""},
            {"id": "OUT", "x": length, "y": 0, "z": 0, "kind": "exit", "area": None, "seats": 0, "label": ""},
        ],
        "edges": [{"id": "Z->OUT", "from": "Z", "to": "OUT", "kind": "door", "width": door_width, "n": 1,
                   "length": length, "use": "normal", "blockable": True, "group": "door", "note": ""}],
        "exits": ["OUT"],
    }


def instant_release(H=600):
    r = np.zeros(H)
    r[0] = 1.0
    return r


def test_sfpe_capacity_values():
    # doorway: Fsm = 1.40 / (4*0.266) = 1.316 p/s/m; effective width = 1.2 - 2*0.15
    assert capacity(P, "door", 1.2) == pytest.approx(1.316 * 0.9, rel=1e-3)
    assert free_speed(P, "corridor") == pytest.approx(0.85 * 1.40)
    assert capacity(P, "door", 1.0, n=3) == pytest.approx(3 * capacity(P, "door", 1.0))


def test_single_door_matches_analytic():
    """N people released at t=0 through one door: T = N / c + transit (within one step per edge)."""
    net = Network(toy_room(), P)
    ps = PathSet.build(net)
    frac = np.ones(len(ps.commodities))
    res = simulate(net, ps, frac, instant_release(), 1.0, 600)
    c, tau = net.cap[0], net.tau[0]
    expected = 200 / c + tau
    assert res.T == pytest.approx(expected, rel=0.05, abs=2.0)
    q = solve_quickest(net, instant_release(), 1.0, 600)
    assert q.T == pytest.approx(res.T, abs=1.0)


def test_spillback_never_speeds_up():
    net = make_network("emergency")
    ps = PathSet.build(net)
    rel = premovement_release(P, "concert", 1.0, 420)
    frac = build_policy(ps, {"habit": 1}, rel, 1.0, 420, P["premovement"]["concert"], iters=5)
    with_sb = simulate(net, ps, frac, rel, 1.0, 420, spillback=True)
    without = simulate(net, ps, frac, rel, 1.0, 420, spillback=False)
    assert with_sb.T >= without.T - 1e-9


@pytest.fixture(scope="module")
def concert():
    net = make_network("emergency")
    ps = PathSet.build(net)
    rel = premovement_release(P, "concert", 1.0, 420)
    return net, ps, rel


def test_people_are_conserved(concert):
    net, ps, rel = concert
    frac = build_policy(ps, {"nearest": 1}, rel, 1.0, 420, P["premovement"]["concert"], iters=5)
    res = simulate(net, ps, frac, rel, 1.0, 420)
    assert sum(res.exit_totals.values()) == pytest.approx(net.N, abs=0.6)
    for i, z in enumerate(net.zones):  # each zone's shares sum to one
        assert frac[ps.by_zone[z]].sum() == pytest.approx(1.0)


def test_lp_optimum_is_a_lower_bound(concert):
    """T* (quickest flow) <= every simulated routing, and >= the analytic bounds."""
    net, ps, rel = concert
    q = solve_quickest(net, rel, 1.0, 300)
    lb = lower_bounds(net, "concert")
    assert q.T >= lb["T_lower_bound"] - 1.0
    for spec in ({"ue": 1}, {"habit": 1}, {"assigned": 0.6, "habit": 0.25, "nearest": 0.15}):
        frac = build_policy(ps, spec, rel, 1.0, 420, P["premovement"]["concert"], iters=8)
        res = simulate(net, ps, frac, rel, 1.0, 420)
        assert q.T <= res.T + 1e-9, spec


def test_unknown_obstruction_costs_at_least_as_much_as_known():
    B = load_building()
    blk = one_leaf_blocked(B, "C090->FOYER")
    rel = premovement_release(P, "concert", 1.0, 420)
    T = {}
    for known in (True, False):
        net = make_network("emergency", blk, known=known)
        ps = PathSet.build(net)
        plan = net if known else make_network("emergency")
        frac = build_policy(PathSet(plan, ps.commodities, ps.by_zone), {"nearest": 1}, rel, 1.0, 420,
                            P["premovement"]["concert"], iters=8)
        T[known] = simulate(net, ps, frac, rel, 1.0, 420).T
    assert math.isfinite(T[False])
    assert T[False] >= T[True] - 2.0


def test_dead_end_obstruction_walks_back():
    """Blocking the only door out of a vestibule must not trap anyone: people retrace their path."""
    B = load_building()
    blk = one_leaf_blocked(B, "V_W->OUT_NW")
    net = make_network("emergency", blk, known=False)
    ps = PathSet.build(net)
    rel = premovement_release(P, "concert", 1.0, 420)
    frac = np.zeros(len(ps.commodities))
    west = {"O_W", "B_W", "Z_M1"}             # west-side zones head for the (blocked) NW exit,
    for z in net.zones:                       # everyone else takes their fastest path
        ks = [k for k in ps.by_zone[z] if ps.commodities[k].exit == "OUT_NW"] if z in west else ps.by_zone[z]
        frac[min(ks, key=ps.tau)] = 1.0
    res = simulate(net, ps, frac, rel, 1.0, 900)  # naive routing is slow; we only check nobody is trapped
    assert math.isfinite(res.T)
    assert res.exit_totals["OUT_NW"] == pytest.approx(0.0)
