"""Run Model I and write results/*.csv and figures/*.png.

    python scripts/run_model1.py network     # building figure + edge/capacity table
    python scripts/run_model1.py baseline    # Q2 + Q3 baselines, policy comparison
    python scripts/run_model1.py blocked     # Q2/Q3: every single obstruction, known vs unknown
    python scripts/run_model1.py all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bailey.flow.model1 import REFERENCE_POLICIES, load_scenario, run_scenario, summary_rows  # noqa: E402
from bailey.flow.network import load_building, make_network, one_leaf_blocked  # noqa: E402
from bailey.params import ROOT  # noqa: E402
from bailey import plotting as pl  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "figures"
BASELINE_SCENARIOS = ["concert_cm", "concert_nocm", "lecture"]
BLOCKED_SCENARIOS = ["concert_cm", "lecture"]


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


# ---------------------------------------------------------------- network figure + table
def cmd_network() -> None:
    import matplotlib.pyplot as plt

    pl.apply_style()
    net = make_network("emergency")
    write_csv(RES / "model1_edges.csv", net.summary_table())
    B = net.building
    nodes = {n["id"]: n for n in B["nodes"]}
    style = {  # role -> (colour, label)
        "door": (pl.SERIES[0], "Door / exit"), "corridor": (pl.SERIES[2], "Corridor"),
        "stair": (pl.SERIES[1], "Stair"), "aisle": ("#b9b8b3", "Seating aisle"),
    }
    role = lambda k: "stair" if "stair" in k else ("aisle" if k == "aisle" else k if k in style else "door")  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1]})
    for ax, floor, title in ((axes[0], 0.0, "First floor: orchestra, mezzanine, corridor"),
                             (axes[1], 5.0, "Balcony and its stairs")):
        seen = set()
        for i, e in enumerate(net.edges):
            a, b = nodes[e["from"]], nodes[e["to"]]
            if floor == 0.0 and (a["z"] > 0 and b["z"] > 0):
                continue
            if floor == 5.0 and not (a["z"] > 0 or b["z"] > 0):
                continue
            r = role(e["kind"])
            col, lab = style[r]
            lw = 0.6 + 0.9 * min(net.cap[i], 10) / 2
            ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=col, lw=lw,
                    ls="--" if e["use"] == "emergency" else "-", solid_capstyle="round",
                    label=lab if r not in seen else None, zorder=2)
            seen.add(r)
        for n in B["nodes"]:
            on = (n["z"] > 0) == (floor > 0) or n["kind"] in ("exit",) or n["id"].startswith(("V_", "C0", "C1"))
            if not on:
                continue
            if n["kind"] == "zone":
                ax.scatter(n["x"], n["y"], s=n["seats"] * 1.6, color="#d9d8d3", edgecolor="white", lw=2, zorder=3)
                ax.text(n["x"], n["y"], f"{n['id']}\n{n['seats']}", ha="center", va="center", fontsize=6.5,
                        color=pl.TEXT, zorder=4)
            elif n["kind"] == "exit":
                ax.scatter(n["x"], n["y"], s=70, marker="s", color=pl.TEXT, zorder=4)
                ax.annotate(n["id"].replace("OUT_", ""), (n["x"], n["y"]), xytext=(4, -10),
                            textcoords="offset points", fontsize=7, color=pl.TEXT)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left")
        ax.set_xlabel("x [m]  (west <-> east)")
        ax.grid(False)
    axes[0].set_ylabel("y [m]  (foyer <-> stage)")
    axes[0].plot([], [], color=pl.TEXT_2, ls="--", label="Emergency use only")
    axes[0].legend(loc="upper center", ncol=5, bbox_to_anchor=(1.05, -0.12))
    fig.suptitle("Bailey Hall as a flow network (line width ~ capacity)", x=0.01, y=1.0, ha="left",
                 fontsize=12, fontweight="bold", color=pl.TEXT)
    fig.savefig(FIG / "model1_network.png")
    plt.close(fig)
    print("  wrote figures/model1_network.png")


# ---------------------------------------------------------------- baseline
def group_queue(run, label: str, group: str) -> np.ndarray:
    r = run.sims[label]
    E = len(run.net.edges)
    idx = [i for i in range(E) if run.net.edges[i]["group"] == group]
    return r.queue_hist[idx].sum(0)


def cmd_baseline() -> None:
    import matplotlib.pyplot as plt

    pl.apply_style()
    rows, runs, bounds = [], {}, {}
    for name in BASELINE_SCENARIOS:
        t0 = time.time()
        scn = load_scenario(name)
        run = run_scenario(scn, policies=REFERENCE_POLICIES)
        runs[name] = run
        rows += summary_rows(run)
        bounds[name] = {k: v for k, v in run.bounds.items()}
        Ts = {k: v.T for k, v in run.sims.items()}
        print(f"  {name:13s} T*={run.quickest.T:.0f}s  " + "  ".join(f"{k}={v:.0f}" for k, v in Ts.items())
              + f"   ({time.time() - t0:.0f}s)")
    write_csv(RES / "model1_baseline.csv", rows)
    (RES / "model1_bounds.json").write_text(json.dumps(bounds, indent=1, default=float))

    N = runs["concert_cm"].net.N
    # Fig: concert policy comparison
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    run = runs["concert_cm"]
    order = ["T*", "ue", "assigned", "scenario", "nearest", "habit"]
    for lab in order:
        if lab == "T*":
            y = N - run.quickest.cumulative; T = run.quickest.T; ls = ":"
        else:
            y = run.sims[lab].in_building(N); T = run.sims[lab].T; ls = "-"
        t = np.arange(len(y)) * run.sims["scenario"].dt
        name = pl.POLICY_LABEL[lab] + (" (6 CMs, 60% comply)" if lab == "scenario" else "")
        ax.plot(t, y, color=pl.POLICY_COLOR[lab], ls=ls, label=f"{name}: {T:.0f} s")
    ax.set_xlim(0, 300); ax.set_ylim(0, N * 1.02)
    pl.egress_limit_line(ax)
    ax.set_xlabel("Time since alarm [s]"); ax.set_ylabel("People still in the building")
    ax.set_title("Q2 concert evacuation: routing matters more than capacity", loc="left")
    ax.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    fig.savefig(FIG / "model1_concert_policies.png"); plt.close(fig)

    # Fig: scenario comparison (Q2 vs Q3)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, name in enumerate(BASELINE_SCENARIOS):
        r = runs[name].sims["scenario"]
        y = r.in_building(N)
        ax.plot(np.arange(len(y)) * r.dt, y, color=pl.SERIES[i],
                label=f"{runs[name].scenario['label']}: {r.T:.0f} s")
    ax.set_xlim(0, 300); ax.set_ylim(0, N * 1.02)
    pl.egress_limit_line(ax)
    ax.set_xlabel("Time since alarm [s]"); ax.set_ylabel("People still in the building")
    ax.set_title("Concert vs lecture (scenario behaviour mixes)", loc="left")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    fig.savefig(FIG / "model1_scenarios.png"); plt.close(fig)

    # Fig: bottleneck queues in the concert scenario mix
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    groups = [("access", "Orchestra -> mezzanine access"), ("mezz_door", "9 mezzanine doors"),
              ("foyer_inner", "3 corridor -> foyer doors"), ("portico", "4 portico doors")]
    for i, (g, lab) in enumerate(groups):
        q = group_queue(run, "scenario", g)
        ax.plot(np.arange(len(q)), q, color=pl.SERIES[i], label=lab)
    ax.set_xlim(0, run.sims["scenario"].T + 5)
    ax.set_xlabel("Time since alarm [s]"); ax.set_ylabel("People queueing")
    ax.set_title("Where the concert crowd waits (Crowd-Manager scenario)", loc="left")
    ax.legend(loc="upper right")
    fig.savefig(FIG / "model1_queues.png"); plt.close(fig)

    # Fig: exit usage per scenario (grouped horizontal bars)
    exits = list(runs["concert_cm"].net.exits)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    h = 0.26
    for i, name in enumerate(BASELINE_SCENARIOS):
        vals = [runs[name].sims["scenario"].exit_totals[x] for x in exits]
        ax.barh(np.arange(len(exits)) + (i - 1) * h, vals, height=h - 0.03, color=pl.SERIES[i],
                label=runs[name].scenario["label"])
    ax.set_yticks(np.arange(len(exits)), [x.replace("OUT_", "") for x in exits])
    ax.invert_yaxis()
    ax.set_xlabel("People leaving through exit"); ax.grid(axis="y", visible=False)
    ax.set_title("Exit usage", loc="left"); ax.legend(loc="lower right")
    fig.savefig(FIG / "model1_exit_usage.png"); plt.close(fig)
    print("  wrote figures/model1_{concert_policies,scenarios,queues,exit_usage}.png")


# ---------------------------------------------------------------- blocked exits
def cmd_blocked(with_lp: bool) -> None:
    import matplotlib.pyplot as plt

    pl.apply_style()
    B = load_building()
    targets = [e["id"] for e in B["edges"] if e["blockable"]]
    rows = []
    for name in BLOCKED_SCENARIOS:
        scn = load_scenario(name)
        base = run_scenario(scn, with_lp=with_lp, with_bounds=False)
        rows += summary_rows(base)
        T0 = base.sims["scenario"].T
        print(f"  {name}: baseline {T0:.0f}s; sweeping {len(targets)} obstructions")
        for eid in targets:
            blk = one_leaf_blocked(B, eid)
            msg = []
            for known in (True, False):
                if not known and blk[eid] > 0:
                    continue  # partial blockage (one leaf of a multi-leaf door): nothing to discover
                run = run_scenario(scn, blocked=blk, known=known, with_lp=with_lp and known, with_bounds=False)
                rows += summary_rows(run)
                msg.append(f"{'known' if known else 'unknown'}={run.sims['scenario'].T:.0f}s")
            print(f"    {eid:22s} " + "  ".join(msg))
    write_csv(RES / "model1_blocked.csv", rows)
    plot_blocked(rows)


def plot_blocked(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    for name in BLOCKED_SCENARIOS:
        sc = [r for r in rows if r["scenario"] == name and r["policy"] == "scenario"]
        T0 = next(r["T"] for r in sc if r["blocked"] == "none")
        items = {}
        for r in sc:
            if r["blocked"] == "none":
                continue
            eid = r["blocked"].split("x")[0]
            items.setdefault(eid, {})["known" if r["known"] else "unknown"] = r["T"]
        order = sorted(items, key=lambda k: -max(items[k].values()))
        fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(order) + 1.4))
        y = np.arange(len(order))
        kn = np.array([items[k].get("known", np.nan) for k in order])
        un = np.array([items[k].get("unknown", np.nan) for k in order])
        for i in range(len(order)):
            vals = [v for v in (kn[i], un[i]) if np.isfinite(v)]
            ax.plot([min(vals), max(vals)], [i, i], color=pl.GRID, lw=2, zorder=1)
        ax.scatter(kn, y, s=36, color=pl.SERIES[0], label="Obstruction known at t = 0 (CMs redirect)", zorder=3,
                   edgecolor="white", lw=1.5)
        ax.scatter(un, y, s=36, color=pl.SERIES[1], label="Discovered on arrival (+10 s, re-route)", zorder=3,
                   edgecolor="white", lw=1.5)
        ax.axvline(T0, color=pl.REFERENCE, lw=1, ls=":")
        ax.axvline(180, color=pl.REFERENCE, lw=1, ls="--")
        ax.text(T0, -1.2, f"no obstruction {T0:.0f} s", fontsize=7.5, color=pl.TEXT_2, ha="center")
        ax.text(180, -1.2, "180 s", fontsize=7.5, color=pl.TEXT_2, ha="center")
        ax.set_yticks(y, order, fontsize=7.5)
        ax.set_ylim(len(order) - 0.4, -1.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Evacuation time [s]")
        ax.set_title(f"Single obstructions: {load_scenario(name)['label']}", loc="left")
        ax.legend(loc="lower right", fontsize=7.5)
        fig.savefig(FIG / f"model1_blocked_{name}.png"); plt.close(fig)
        print(f"  wrote figures/model1_blocked_{name}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["network", "baseline", "blocked", "all"])
    ap.add_argument("--no-lp", action="store_true", help="skip T* LPs in the obstruction sweep (faster)")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    if a.what in ("network", "all"):
        cmd_network()
    if a.what in ("baseline", "all"):
        cmd_baseline()
    if a.what in ("blocked", "all"):
        cmd_blocked(with_lp=not a.no_lp)


if __name__ == "__main__":
    main()
