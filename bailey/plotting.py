"""Shared figure style for the paper (static matplotlib).

Categorical colours follow a fixed, CVD-validated order (dataviz reference palette); a colour always
follows its entity (e.g. a routing policy keeps its colour in every figure), never its rank.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
REFERENCE = "#8a8984"   # neutral ink for reference lines (180 s limit, bounds)

# fixed entity -> colour assignments, used across all figures
POLICY_COLOR = {
    "T*": REFERENCE, "ue": SERIES[0], "assigned": SERIES[2], "scenario": SERIES[1],
    "nearest": SERIES[6], "habit": SERIES[3],
}
POLICY_LABEL = {
    "T*": "System optimum T* (LP bound)", "ue": "Fully informed (user equilibrium)",
    "assigned": "LP-assigned routing", "scenario": "Scenario mix", "nearest": "Nearest exit",
    "habit": "Habit: all via foyer",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0,
        "font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def egress_limit_line(ax, t: float = 180.0, label: str = "180 s code limit") -> None:
    ax.axvline(t, color=REFERENCE, lw=1.0, ls="--", zorder=1)
    ax.text(t + 3, ax.get_ylim()[1] * 0.10, label, color=TEXT_2, fontsize=8, va="bottom")
