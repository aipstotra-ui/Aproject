"""Load data/params.yaml and turn edge geometry into capacities and free-walking speeds (SFPE hydraulic model)."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "data" / "params.yaml"
BUILDING_PATH = ROOT / "data" / "building.json"


@lru_cache(maxsize=4)
def load_params(path: str | None = None) -> dict:
    with open(path or PARAMS_PATH) as f:
        return yaml.safe_load(f)


def _kind(params: dict, kind: str) -> dict:
    return params["edge_kinds"][kind]


def k_value(params: dict, kind: str) -> float:
    return params["hydraulic"]["k"][_kind(params, kind)["k"]]


def max_specific_flow(params: dict, kind: str) -> float:
    """Fsm = k / (4a) [persons / s / m effective width]."""
    return k_value(params, kind) / (4.0 * params["hydraulic"]["a"])


def effective_width(params: dict, kind: str, width: float, n: int = 1) -> float:
    layer = params["boundary_layer"][_kind(params, kind)["layer"]]
    return n * max(width - 2.0 * layer, 0.0)


def capacity(params: dict, kind: str, width: float, n: int = 1) -> float:
    """Maximum flow [persons/s] through an element. Virtual 'hesitation' edges are unconstrained."""
    if kind == "hesitation":
        return math.inf
    return max_specific_flow(params, kind) * effective_width(params, kind, width, n)


def free_speed(params: dict, kind: str) -> float:
    """Unimpeded walking speed [m/s] = 0.85 k (times an optional kind-specific factor)."""
    kd = _kind(params, kind)
    return params["hydraulic"]["free_speed_factor"] * k_value(params, kind) * kd.get("speed_factor", 1.0)


def premovement_release(params: dict, profile: str, dt: float, horizon: float) -> np.ndarray:
    """Fraction of an audience that starts moving in each time step [t, t+dt).

    Lognormal(median, sigma) truncated below at t_min and above at quantile q_cap
    (everyone left at the cap starts then). Returns array of length ceil(horizon/dt) summing to 1.
    """
    from scipy.stats import lognorm

    p = params["premovement"][profile]
    dist = lognorm(s=p["sigma"], scale=p["median"])
    t_cap = float(dist.ppf(p["q_cap"]))
    n = int(math.ceil(horizon / dt))
    edges = np.arange(n + 1) * dt
    cdf = dist.cdf(np.clip(edges, p["t_min"], t_cap))
    cdf = (cdf - dist.cdf(p["t_min"])) / (dist.cdf(t_cap) - dist.cdf(p["t_min"]))
    cdf[edges >= t_cap] = 1.0
    frac = np.diff(cdf)
    return frac / frac.sum()


def premovement_stats(params: dict, profile: str) -> dict:
    from scipy.stats import lognorm

    p = params["premovement"][profile]
    dist = lognorm(s=p["sigma"], scale=p["median"])
    return {"median": p["median"], "t_min": p["t_min"], "t_cap": float(dist.ppf(p["q_cap"]))}
