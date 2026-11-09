"""
Question 4a: can temporary barriers in the corridor make the evacuation faster?

    python barriers.py

The foyer is the busiest route, so we try a barrier across the corridor next to it,
always on both sides of the hall at once (the hall is symmetrical):
    layout A: between mezz doors 3 and 4 (and 6 and 7 on the other side)
    layout B: between mezz door 4 and the foyer (and between the foyer and mezz door 6)
A barrier is just a wall: people who come out of a mezzanine door on the far side of it
can't walk along the corridor to the foyer, so they use a side exit (or another mezzanine door).
Everything else is the concert with Crowd Managers (80 % guided), 30 runs per layout.
Results: output/barriers.png, output/barrier_heatmaps.png, output/barriers.csv
"""
import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import simulation as sim

NUMBER_OF_RUNS = 30
TIME_LIMIT = 180
OUTPUT_FOLDER = "output"
CASE = sim.CASES[2]            # concert, emergency, Crowd Managers (80 % guided)

# Where each barrier goes. The hall is round, so a place in the corridor is an angle
# (0 = west end by the stage, 90 = the foyer, 180 = east end). The mezzanine doors 1-9 sit at
# 10, 30, 50, ..., 170 degrees and the 3 foyer doors at 80, 90 and 100 degrees.
BARRIER_SPOTS = {
    "mezz door 3 | 4": 60,     # halfway between door 3 (50) and door 4 (70)
    "mezz door 6 | 7": 120,    # its mirror image
    "mezz door 4 | 5": 75,     # halfway between door 4 (70) and foyer door 1 (80): halfway between
    "mezz door 5 | 6": 105,    #   doors 4 and 5 (80) would be ON foyer door 1, so it can't go there
}
LAYOUTS = [
    ("No barriers", []),
    ("A: between mezz doors 3|4 and 6|7", ["mezz door 3 | 4", "mezz door 6 | 7"]),
    ("B: between mezz doors 4|5 and 5|6", ["mezz door 4 | 5", "mezz door 5 | 6"]),
]
COLOURS = ["#6250d6", "#1baf7a", "#eb6834"]


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    data = sim.load_flow_data()
    rows = []
    results = []
    for name, spots in LAYOUTS:
        hall = sim.setup([BARRIER_SPOTS[spot] for spot in spots])
        times = []
        exits = {e: 0 for e in sim.EXIT_NAMES}
        density = np.zeros(hall["grid"].shape)
        finish = {}
        print(f"\n{name}")
        for run in range(NUMBER_OF_RUNS):            # run i uses seed i and data row i, for every layout
            rates = sim.rates_for_run(data, CASE["data_case"], run)
            people, times_occupied, steps, _ = sim.run_once(run, hall, CASE, rates)
            last_out, doorways = sim.calculated_escape_time(people, hall, rates)
            times.append(last_out)
            density += times_occupied / steps / NUMBER_OF_RUNS
            for p in people:
                if p["exit_used"] is not None:
                    exits[p["exit_used"]] += 1
            for doorway, info in doorways.items():
                finish.setdefault(doorway, []).append(info["finish"])
            rows.append({"layout": name, "run": run, "calculated_time_s": round(last_out, 1)})
            print(f"  run {run + 1:2d}: {last_out:6.1f} s")
        results.append({"name": name, "hall": hall, "times": times, "density": density,
                         "exits": {e: n / NUMBER_OF_RUNS for e, n in exits.items()},
                         "finish": {d: statistics.mean(v) for d, v in finish.items()}})

    # ---- summary ----
    print("\n================ summary ================")
    for r in results:
        t = r["times"]
        under = sum(1 for x in t if x <= TIME_LIMIT)
        slowest = sorted(r["finish"], key=r["finish"].get, reverse=True)[:3]
        print(f"\n{r['name']}")
        print(f"  last person out : {statistics.mean(t):.1f} s +- {statistics.stdev(t):.1f}"
              f" (fastest {min(t):.1f}, slowest {max(t):.1f}), within {TIME_LIMIT} s in {under}/{NUMBER_OF_RUNS}")
        print("  doorways that finish last: " + ", ".join(f"{d} {r['finish'][d]:.0f} s" for d in slowest))
        print("  people per exit : " + ", ".join(f"{e} {n:.0f}" for e, n in r["exits"].items()))

    with open(os.path.join(OUTPUT_FOLDER, "barriers.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---- figure 1: escape time per layout, one dot per run ----
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for row, (r, colour) in enumerate(zip(results, COLOURS)):
        wobble = np.linspace(-0.12, 0.12, len(r["times"]))
        ax.scatter(r["times"], row + wobble, color=colour, s=22, alpha=0.8)
        mean_time = statistics.mean(r["times"])
        under = sum(1 for x in r["times"] if x <= TIME_LIMIT)
        ax.plot([mean_time, mean_time], [row - 0.28, row + 0.28], color="#222222", linewidth=2)
        ax.text(mean_time, row + 0.33, f"average {mean_time:.0f} s, {under}/{NUMBER_OF_RUNS} within 180 s",
                ha="center", fontsize=9)
    ax.axvline(TIME_LIMIT, color="#c0392b", linestyle="--", linewidth=1.2)
    ax.set_yticks(range(len(results)), [r["name"] for r in results])
    ax.set_ylim(-0.6, len(results) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("seconds until the last person is out (calculated)")
    ax.set_title("Concert with Crowd Managers: effect of corridor barriers", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_FOLDER, "barriers.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: where people crowd, one map per layout (barriers show up as walls) ----
    extent = [sim.X_MIN, sim.X_MAX, sim.Y_MIN, sim.Y_MAX]
    top = max(float(r["density"].max()) for r in results)
    fig, axes = plt.subplots(1, len(results), figsize=(20, 4.6))
    for ax, r in zip(axes, results):
        grid = r["hall"]["grid"]
        ax.imshow(grid == sim.WALL, cmap=ListedColormap(["white", "#3b3f45"]), extent=extent, interpolation="nearest")
        picture = ax.imshow(np.ma.masked_where(grid == sim.WALL, r["density"]), cmap="YlOrRd", vmin=0, vmax=top,
                            extent=extent, interpolation="nearest")
        for spot in dict(LAYOUTS)[r["name"]]:          # mark the barriers in blue
            angle = BARRIER_SPOTS[spot]
            x1, y1 = sim.point_on_circle(sim.MEZZ_WALL_END, angle)
            x2, y2 = sim.point_on_circle(sim.CORRIDOR_END, angle)
            ax.plot([x1, x2], [y1, y2], color="#1f77d0", linewidth=3)
        ax.set_title(r["name"], loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(top=0.86, right=0.9, wspace=0.05)
    bar = fig.colorbar(picture, cax=fig.add_axes([0.92, 0.15, 0.012, 0.65]))
    bar.set_label("share of time the square was occupied")
    fig.suptitle("Where people crowd with and without barriers (blue = barrier)", x=0.02, y=0.98, ha="left",
                 fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(OUTPUT_FOLDER, "barrier_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUTPUT_FOLDER}/barriers.png, barrier_heatmaps.png and barriers.csv")


if __name__ == "__main__":
    main()
