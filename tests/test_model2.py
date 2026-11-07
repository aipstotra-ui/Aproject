"""Verification of Model II (cellular automaton)."""
import numpy as np
import pytest

from bailey.flow.network import load_building, one_leaf_blocked
from bailey.params import load_params, max_specific_flow
from bailey.sim.calibrate import door_flow
from bailey.sim.grid import EXITS, build_grid
from bailey.sim.model2 import Model2, ca_blocked

P = load_params()


@pytest.fixture(scope="module")
def m2():
    return Model2()


def test_grid_is_connected_and_matches_model1(m2):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    g = m2.grid
    E, _ = g.edges()
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(g.n, g.n))
    assert connected_components(A, directed=False)[0] == 1
    seats = {n["id"]: n["seats"] for n in load_building()["nodes"] if n["kind"] == "zone"}
    for z, n in seats.items():                      # every Model I zone fits in its CA area
        assert len(g.zone_cells[z]) >= n, z
    blockable = [e["id"] for e in load_building()["edges"] if e["blockable"]]
    for eid in blockable:                           # every Model I obstruction maps onto the grid
        assert ca_blocked(g, one_leaf_blocked(load_building(), eid)), eid


def test_door_flow_calibrated_to_sfpe():
    target = max_specific_flow(P, "door") * P["ca"]["cell"]
    assert door_flow(P, door_cells=3, people=250, seeds=4) == pytest.approx(target, rel=0.08)


def test_everyone_gets_out_and_is_counted_once(m2):
    r = m2.run("lecture", runs=2)
    for run in r.runs:
        assert run.remaining == 0
        assert np.isfinite(run.exit_time).all()
        assert (run.exit_id >= 0).all() and len(run.exit_id) == 1329
        assert run.exit_time.min() >= run.start_time.min()


def test_no_gridlock_in_counterflow(m2):
    """The unguided lecture mix creates opposing streams in the corridor; the exchange rule must
    prevent permanent jams (this failed before the rule was added)."""
    r = m2.run("lecture", runs=6)
    assert np.isfinite(r.T).all()


def test_unknown_obstruction_is_discovered(m2):
    B = load_building()
    r = m2.run("concert_cm", blocked=one_leaf_blocked(B, "C090->FOYER"), known=False, runs=2)
    assert np.isfinite(r.T).all()
    known = m2.run("concert_cm", blocked=one_leaf_blocked(B, "C090->FOYER"), known=True, runs=2)
    assert np.isfinite(known.T).all()


def test_blocked_exit_is_never_used(m2):
    B = load_building()
    r = m2.run("concert_cm", blocked=one_leaf_blocked(B, "V_E->OUT_NE"), known=False, runs=2)
    ne = EXITS.index("OUT_NE")
    for run in r.runs:
        assert (run.exit_id != ne).all()
