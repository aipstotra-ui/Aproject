"""Run Model II (cellular automaton) and write results/model2_*.csv and figures/model2_*.png.

    python scripts/run_model2.py calibrate   # friction mu vs SFPE door flow; width check
    python scripts/run_model2.py crossval    # Model I vs Model II on identical behaviour mixes
    python scripts/run_model2.py baseline    # Q2/Q3 Monte Carlo, egress bands, snapshots, heat map
    python scripts/run_model2.py blocked     # key obstructions, known vs unknown, vs Model I
    python scripts/run_model2.py all
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bailey import plotting as pl  # noqa: E402
from bailey.flow.model1 import REFERENCE_POLICIES, load_scenario, run_scenario  # noqa: E402
from bailey.flow.network import load_building, one_leaf_blocked  # noqa: E402
from bailey.params import ROOT, load_params, max_specific_flow  # noqa: E402
from bailey.sim.calibrate import door_flow  # noqa: E402
from bailey.sim.grid import EXITS  # noqa: E402
from bailey.sim.model2 import Model2  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
SCENARIOS = ["concert_cm", "concert_nocm", "lecture"]
KEY_OBSTRUCTIONS = ["C090->FOYER", "C080->FOYER", "A4->M6", "M4->C070", "V_E->OUT_NE", "SE_E->V_E",
                    "BS_W->C065", "C050->OUT_P2", "FOYER->OUT_FRONT"]


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------------------- calibrate
def cmd_calibrate() -> None:
    import matplotlib.pyplot as plt

    pl.apply_style()
    P = load_params()
    target = max_specific_flow(P, "door") * P["ca"]["cell"]
    rows = []
    for mu in [0.0, 0.2, 0.4, 0.5, 0.6, 0.67, 0.75, 0.85]:
        Q = copy.deepcopy(P); Q["ca"]["friction"] = mu
        f = door_flow(Q, 3, 300, 8)
        rows.append({"test": "friction", "mu": mu, "door_cells": 3, "flow_per_cell": f, "flow_per_m": f / Q["ca"]["cell"]})
        print(f"  mu={mu:.2f}: {f / Q['ca']['cell']:.3f} p/s/m")
    for k in [1, 2, 3, 4, 6, 8]:
        f = door_flow(P, k, 300, 8)
        rows.append({"test": "width", "mu": P["ca"]["friction"], "door_cells": k, "flow_per_cell": f,
                     "flow_per_m": f / P["ca"]["cell"]})
        print(f"  width {k * P['ca']['cell']:.1f} m: {f / P['ca']['cell']:.3f} p/s/m")
    write_csv(RES / "model2_calibration.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    fr = [r for r in rows if r["test"] == "friction"]
    axes[0].plot([r["mu"] for r in fr], [r["flow_per_m"] for r in fr], marker="o", color=pl.SERIES[0], ms=5)
    axes[0].axhline(target / P["ca"]["cell"], color=pl.REFERENCE, ls="--", lw=1)
    axes[0].text(0.02, target / P["ca"]["cell"] + 0.03, "SFPE max specific flow 1.32 p/s/m", fontsize=8, color=pl.TEXT_2)
    axes[0].axvline(P["ca"]["friction"], color=pl.REFERENCE, ls=":", lw=1)
    axes[0].set_xlabel("Friction parameter $\\mu$"); axes[0].set_ylabel("Door flow [persons/s per m]")
    axes[0].set_title("Calibrating friction", loc="left")
    wd = [r for r in rows if r["test"] == "width"]
    axes[1].plot([r["door_cells"] * P["ca"]["cell"] for r in wd], [r["flow_per_m"] for r in wd], marker="o",
                 color=pl.SERIES[0], ms=5)
    axes[1].axhline(target / P["ca"]["cell"], color=pl.REFERENCE, ls="--", lw=1)
    axes[1].set_ylim(0, 1.8)
    axes[1].set_xlabel("Effective door width [m]"); axes[1].set_title(f"Check across widths ($\\mu$ = {P['ca']['friction']})", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "model2_calibration.png"); plt.close(fig)
    print("  wrote figures/model2_calibration.png")


# ------------------------------------------------------------------------------------- cross-validation
def cmd_crossval(runs: int) -> None:
    import matplotlib.pyplot as plt

    pl.apply_style()
    m = Model2()
    rows = []
    configs = [("concert_cm", lab, spec) for lab, spec in REFERENCE_POLICIES.items()]
    configs += [(s, "scenario", None) for s in SCENARIOS]
    m1_cache = {}
    for scn_name, lab, spec in configs:
        scn = load_scenario(scn_name)
        if scn_name not in m1_cache:
            m1_cache[scn_name] = run_scenario(scn, policies=REFERENCE_POLICIES, with_lp=False, with_bounds=False)
        T1 = m1_cache[scn_name].sims[lab].T
        r = m.run(scn, routing=spec, crowd_managers=False, runs=runs, variant="crossval")
        s = r.summary()
        rows.append({"scenario": scn_name, "policy": lab, "model1_T": T1, **{k: s[k] for k in ("T_mean", "T_sd", "T_p05", "T_p95")}})
        print(f"  {scn_name:13s} {lab:9s} Model I {T1:6.1f}   Model II {s['T_mean']:6.1f} +- {s['T_sd']:.1f}")
    write_csv(RES / "model2_crossval.csv", rows)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lim = [80, 300]
    ax.plot(lim, lim, color=pl.REFERENCE, lw=1, ls="--", zorder=1)
    ax.text(108, 102, "Model II = Model I", ha="center", va="top", fontsize=8, color=pl.TEXT_2, rotation=45,
            rotation_mode="anchor")
    for r in rows:
        col = pl.POLICY_COLOR.get(r["policy"], pl.SERIES[1]) if r["scenario"] == "concert_cm" else pl.SERIES[1]
        mk = "o" if r["scenario"] == "concert_cm" else {"concert_nocm": "s", "lecture": "D"}[r["scenario"]]
        ax.errorbar(r["model1_T"], r["T_mean"], yerr=[[r["T_mean"] - r["T_p05"]], [r["T_p95"] - r["T_mean"]]],
                    fmt=mk, color=col, ms=7, mec="white", mew=1.5, elinewidth=1.2, capsize=0, zorder=3)
        name = (pl.POLICY_LABEL.get(r["policy"], r["policy"]) if r["scenario"] == "concert_cm"
                else load_scenario(r["scenario"])["label"])
        if r["scenario"] == "concert_cm" and r["policy"] == "scenario":
            name = "Concert mix (no CM agents)"
        below = r["policy"] == "ue" and r["scenario"] == "concert_cm"
        if below:
            name = "Fully informed (UE)"
        ax.annotate(name, (r["model1_T"], r["T_mean"]), xytext=(-4, -16) if below else (7, -3),
                    textcoords="offset points", fontsize=7.5, color=pl.TEXT, ha="right" if below else "left")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("Model I (fluid network) evacuation time [s]")
    ax.set_ylabel("Model II (CA) evacuation time [s], mean and 5-95%")
    ax.set_title("Cross-validation on identical behaviour", loc="left")
    fig.savefig(FIG / "model2_crossval.png"); plt.close(fig)
    print("  wrote figures/model2_crossval.png")


# ------------------------------------------------------------------------------------- baseline
def cmd_baseline(runs: int) -> None:
    import matplotlib.pyplot as plt

    pl.apply_style()
    m = Model2()
    rows, res = [], {}
    for name in SCENARIOS:
        t0 = time.time()
        r = m.run(name, runs=runs, snapshot_times=(30, 60, 90, 120))
        res[name] = r
        s = r.summary(); rows.append(s)
        print(f"  {name:13s} T = {s['T_mean']:.1f} +- {s['T_sd']:.1f} s  (5-95%: {s['T_p05']:.0f}-{s['T_p95']:.0f}),"
              f" P(T<=180) = {s['P_T_le_180']:.2f}   ({time.time() - t0:.0f}s)")
    write_csv(RES / "model2_baseline.csv", rows)

    # egress bands, with Model I curve for reference
    m1 = {n: run_scenario(load_scenario(n), with_lp=False, with_bounds=False) for n in SCENARIOS}
    N = 1329
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    t = np.arange(0, 301)
    for i, name in enumerate(SCENARIOS):
        curves = np.array([rr.curve(300)[1] for rr in res[name].runs])
        lo, mid, hi = np.percentile(curves, [5, 50, 95], axis=0)
        ax.fill_between(t, lo, hi, color=pl.SERIES[i], alpha=0.18, lw=0)
        ax.plot(t, mid, color=pl.SERIES[i], label=f"{load_scenario(name)['label']}: {res[name].T.mean():.0f} s (CA)")
        f1 = m1[name].sims["scenario"]
        y1 = f1.in_building(N)
        ax.plot(np.arange(len(y1)) * f1.dt, y1, color=pl.SERIES[i], ls=":", lw=1.4)
    ax.plot([], [], color=pl.TEXT_2, ls=":", lw=1.4, label="Model I (fluid) for the same scenario")
    ax.set_xlim(0, 300); ax.set_ylim(0, N * 1.02)
    pl.egress_limit_line(ax)
    ax.set_xlabel("Time since alarm [s]"); ax.set_ylabel("People still in the building")
    ax.set_title("Model II: median and 5-95% band over Monte Carlo runs", loc="left")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95, fontsize=8)
    fig.savefig(FIG / "model2_egress_bands.png"); plt.close(fig)

    g = m.grid
    # snapshots (concert with Crowd Managers, first run)
    run0 = res["concert_cm"].runs[0]
    times = [30, 60, 90, 120]
    type_col = {0: pl.SERIES[3], 1: pl.SERIES[6], 2: pl.SERIES[2], 3: pl.SERIES[0]}
    type_lab = {0: "Habit (foyer)", 1: "Nearest exit", 2: "Guided by Crowd Managers / plan", 3: "Rational"}
    fig, axes = plt.subplots(2, len(times), figsize=(12, 5.0), gridspec_kw={"height_ratios": [1.35, 0.85]})
    for c, tt in enumerate(times):
        cells, types = run0.snapshots[tt]
        for r_, layer in enumerate((0, 1)):
            ax = axes[r_, c]
            walk = (g.layer == layer)
            ax.scatter(g.x[walk], g.y[walk], s=0.6, color="#e4e3df", marker="s", lw=0)
            for k, col in type_col.items():
                m_ = (types == k) & (g.layer[cells] == layer)
                ax.scatter(g.x[cells[m_]], g.y[cells[m_]], s=2.2, color=col, lw=0,
                           label=type_lab[k] if (r_ == 0 and c == 0) else None)
            ax.set_aspect("equal"); ax.axis("off")
            if r_ == 0:
                ax.set_title(f"t = {tt} s   ({len(cells)} inside)", loc="left", fontsize=9.5)
    axes[0, 0].text(-25, 7.5, "first floor", fontsize=8, color=pl.TEXT_2)
    axes[1, 0].text(-21, 4.5, "balcony", fontsize=8, color=pl.TEXT_2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, markerscale=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(hspace=0.02, wspace=0.04, top=0.88, bottom=0.07)
    fig.suptitle("Model II snapshots: concert with Crowd Managers (one run)", x=0.01, y=0.98, ha="left",
                 fontsize=12, fontweight="bold", color=pl.TEXT)
    fig.savefig(FIG / "model2_snapshots.png"); plt.close(fig)

    # congestion heat map: mean seconds each first-floor cell is occupied after people start moving
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, name in zip(axes, ("concert_cm", "lecture")):
        occ = np.mean([rr.occupancy_time for rr in res[name].runs], axis=0)
        m0 = (g.layer == 0) & ~np.isin(g.region, ["orch", "mezz"])  # circulation space only
        sc = ax.scatter(g.x[m0], g.y[m0], c=occ[m0], s=5, marker="s", cmap="Blues", vmin=0,
                        vmax=np.percentile(occ[m0], 99), lw=0)
        seats = (g.layer == 0) & np.isin(g.region, ["orch", "mezz"])
        ax.scatter(g.x[seats], g.y[seats], s=0.6, color="#ecebe7", marker="s", lw=0)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{load_scenario(name)['label']}", loc="left", fontsize=10)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01); cb.set_label("Seconds occupied", fontsize=8.5)
        cb.outline.set_visible(False)
    fig.suptitle("Where people stand and wait: corridor, foyer and exits (mean over runs)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=pl.TEXT)
    fig.savefig(FIG / "model2_congestion.png"); plt.close(fig)
    print("  wrote figures/model2_{egress_bands,snapshots,congestion}.png")


# ------------------------------------------------------------------------------------- obstructions
def cmd_blocked(runs: int) -> None:
    import matplotlib.pyplot as plt

    pl.apply_style()
    m = Model2()
    B = load_building()
    rows = []
    for name in ("concert_cm", "lecture"):
        base = m.run(name, runs=runs, variant="blocked").summary()
        rows.append(base)
        print(f"  {name}: baseline {base['T_mean']:.0f}s")
        for eid in KEY_OBSTRUCTIONS:
            blk = one_leaf_blocked(B, eid)
            msg = []
            for known in (True, False):
                if not known and blk[eid] > 0:
                    continue
                s = m.run(name, blocked=blk, known=known, runs=runs, variant="blocked").summary()
                rows.append(s)
                msg.append(f"{'known' if known else 'unknown'}={s['T_mean']:.0f}+-{s['T_sd']:.0f}s")
            print(f"    {eid:20s} " + "  ".join(msg))
    write_csv(RES / "model2_blocked.csv", rows)
    plot_blocked(rows)


def plot_blocked(rows: list[dict] | None = None) -> None:
    import matplotlib.pyplot as plt

    pl.apply_style()
    if rows is None:
        rows = [{k: (v if k in ("scenario", "question", "variant", "blocked") else
                     (v == "True") if k == "known" else float(v)) for k, v in r.items()}
                for r in read_csv(RES / "model2_blocked.csv")]
    m1_path = RES / "model1_blocked.csv"
    m1 = {}
    if m1_path.exists():
        for r in read_csv(m1_path):
            if r["policy"] == "scenario":
                m1[(r["scenario"], r["blocked"], r["known"] == "True")] = float(r["T"])
    for name in ("concert_cm", "lecture"):
        sub = [r for r in rows if r["scenario"] == name and r["blocked"] != "none"]
        base = next(r for r in rows if r["scenario"] == name and r["blocked"] == "none")
        eids = list(dict.fromkeys(r["blocked"].split("x")[0] for r in sub))
        eids.sort(key=lambda e: -max(r["T_mean"] for r in sub if r["blocked"].startswith(e)))
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.36 * len(eids) + 1.6), sharey=True, sharex=True)
        for ax, known, title in ((axes[0], True, "Known at t = 0"), (axes[1], False, "Discovered on arrival")):
            for i, e in enumerate(eids):
                r2 = next((r for r in sub if r["blocked"].startswith(e) and r["known"] == known), None)
                if r2 is None:
                    continue
                ax.plot([r2["T_p05"], r2["T_p95"]], [i, i], color=pl.SERIES[0], lw=2, alpha=0.35, solid_capstyle="round")
                ax.scatter(r2["T_mean"], i, color=pl.SERIES[0], s=36, zorder=3, edgecolor="white", lw=1.2,
                           label="Model II (CA): mean, 5-95%" if i == 0 else None)
                t1 = m1.get((name, r2["blocked"], known))
                if t1 is not None:
                    ax.scatter(t1, i, marker="D", color=pl.SERIES[1], s=26, zorder=4, edgecolor="white", lw=1,
                               label="Model I (fluid)" if i == 0 else None)
            ax.axvline(base["T_mean"], color=pl.REFERENCE, ls=":", lw=1)
            ax.axvline(180, color=pl.REFERENCE, ls="--", lw=1)
            ax.set_title(title, loc="left", fontsize=10); ax.grid(axis="y", visible=False)
            ax.set_xlabel("Evacuation time [s]")
        axes[0].set_yticks(range(len(eids)), eids, fontsize=8); axes[0].invert_yaxis()
        axes[1].legend(loc="lower right", fontsize=7.5)
        fig.suptitle(f"Single obstructions, both models: {load_scenario(name)['label']}"
                     f"  (dotted: no obstruction, dashed: 180 s)", x=0.01, ha="left", fontsize=11,
                     fontweight="bold", color=pl.TEXT)
        fig.tight_layout(); fig.savefig(FIG / f"model2_blocked_{name}.png"); plt.close(fig)
        print(f"  wrote figures/model2_blocked_{name}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["calibrate", "crossval", "baseline", "blocked", "plot-blocked", "all"])
    ap.add_argument("--runs", type=int, default=None, help="Monte Carlo runs (default: params ca.runs)")
    a = ap.parse_args()
    runs = a.runs or load_params()["ca"]["runs"]
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    if a.what in ("calibrate", "all"):
        cmd_calibrate()
    if a.what in ("crossval", "all"):
        cmd_crossval(runs)
    if a.what in ("baseline", "all"):
        cmd_baseline(runs)
    if a.what in ("blocked", "all"):
        cmd_blocked(max(10, runs // 3))
    if a.what == "plot-blocked":
        plot_blocked()


if __name__ == "__main__":
    main()
