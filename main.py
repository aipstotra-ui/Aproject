"""
Run the Bailey Hall simulation many times and make the figures.

    python main.py

Results go into the folder output/:
    map.png          the hall as the computer sees it
    people_left.png  people still inside over time, for every run
    heatmap.png      how crowded each square was (the density map)
    results.csv      one line per run
"""
import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")                     # draw into files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import simulation as sim

NUMBER_OF_RUNS = 30        # how many times we repeat the evacuation
TIME_LIMIT = 180           # the fire code says everyone must be out within 180 s
OUTPUT_FOLDER = "output"


def draw_map(hall):
    """Picture of the grid: walls, floor, seats and exits, with the exit names written on."""
    grid = hall["grid"]
    colours = ListedColormap(["#3b3f45", "#f2efe8", "#c9d9ec", "#2e9e5b"])   # wall, floor, seat, exit
    extent = [sim.X_MIN, sim.X_MAX, sim.Y_MIN, sim.Y_MAX]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.imshow(grid, cmap=colours, vmin=0, vmax=3, extent=extent, interpolation="nearest")

    # write each exit's name next to it
    for name in sim.EXIT_NAMES:
        squares = [square for square, exit_name in hall["exit_of_square"].items() if exit_name == name]
        xs = [sim.square_centre(r, c)[0] for r, c in squares]
        ys = [sim.square_centre(r, c)[1] for r, c in squares]
        label_y = np.mean(ys) + 1.5 if name == "FRONT" else np.mean(ys) - 1.3   # FRONT: write it inside the foyer
        ax.text(np.mean(xs), label_y, name, ha="center", va="top", fontsize=9, fontweight="bold",
                color="#1d6b3d")
    # mark where the balcony staircases come down
    for name, how_many, where in sim.BALCONY_STAIRS:
        row, col = sim.stair_landing(grid, where)
        x, y = sim.square_centre(row, col)
        ax.plot(x, y, marker="v", color="#d9822b", markersize=9)
    ax.plot([], [], marker="v", color="#d9822b", linestyle="none", label="balcony stairs arrive here")
    for value, label in [(1, "floor"), (2, "seats"), (3, "exits"), (0, "wall")]:
        ax.plot([], [], marker="s", linestyle="none", markersize=10, color=colours(value), label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    ax.set_title("Bailey Hall as a grid of 0.5 m x 0.5 m squares", loc="left", fontweight="bold")
    ax.set_xlabel("metres (west - east)")
    ax.set_ylabel("metres (foyer - stage)")
    fig.savefig(os.path.join(OUTPUT_FOLDER, "map.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_people_left(all_out_times, total_people):
    """People still inside over time: one thin line per run, a thick line for the average."""
    times = np.arange(0, 301, 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    curves = []
    for out_times in all_out_times:
        out_times = sorted(out_times)
        still_inside = [total_people - np.searchsorted(out_times, t, side="right") for t in times]
        curves.append(still_inside)
        ax.plot(times, still_inside, color="#7aa6d8", linewidth=0.8, alpha=0.6)
    ax.plot(times, np.mean(curves, axis=0), color="#1f4e8c", linewidth=2.5, label="average of all runs")
    ax.plot([], [], color="#7aa6d8", linewidth=0.8, label=f"each of the {len(all_out_times)} runs")
    ax.axvline(TIME_LIMIT, color="#c0392b", linestyle="--", linewidth=1.2)
    ax.text(TIME_LIMIT + 3, total_people * 0.45, "180 s limit", color="#c0392b")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, total_people * 1.03)
    ax.set_xlabel("seconds since the alarm")
    ax.set_ylabel("people still inside")
    ax.set_title("Concert with ushers: how fast the hall empties", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_FOLDER, "people_left.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_heatmap(hall, density):
    """How often each square was occupied (0 = never, 1 = all the time), averaged over all runs."""
    grid = hall["grid"]
    extent = [sim.X_MIN, sim.X_MAX, sim.Y_MIN, sim.Y_MAX]
    shown = np.ma.masked_where(grid == sim.WALL, density)       # don't colour the walls
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.imshow(grid == sim.WALL, cmap=ListedColormap(["white", "#3b3f45"]), extent=extent, interpolation="nearest")
    picture = ax.imshow(shown, cmap="YlOrRd", vmin=0, vmax=max(0.3, float(density.max())), extent=extent,
                        interpolation="nearest")
    bar = fig.colorbar(picture, ax=ax, shrink=0.75)
    bar.set_label("share of time the square was occupied")
    ax.set_title("Where people crowd (average over all runs)", loc="left", fontweight="bold")
    ax.set_xlabel("metres (west - east)")
    ax.set_ylabel("metres (foyer - stage)")
    fig.savefig(os.path.join(OUTPUT_FOLDER, "heatmap.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    hall = sim.setup()
    total_people = sim.ORCHESTRA_PEOPLE + sim.MEZZANINE_PEOPLE + sum(n for _, n, _ in sim.BALCONY_STAIRS)

    evacuation_times = []
    all_out_times = []
    density_sum = np.zeros(hall["grid"].shape)
    exit_totals = {name: 0 for name in sim.EXIT_NAMES}
    rows = []

    for run in range(NUMBER_OF_RUNS):
        people, times_occupied, steps = sim.run_once(run, hall)
        out_times = [p["out_time"] for p in people if p["out_time"] is not None]
        stuck = total_people - len(out_times)
        last_out = max(out_times) if stuck == 0 else float("inf")

        evacuation_times.append(last_out)
        all_out_times.append(out_times)
        density_sum += times_occupied / steps            # share of time steps each square was occupied
        per_exit = {name: 0 for name in sim.EXIT_NAMES}
        for p in people:
            if p["exit_used"] is not None:
                per_exit[p["exit_used"]] += 1
                exit_totals[p["exit_used"]] += 1
        rows.append({"run": run, "evacuation_time_s": round(last_out, 1), "people_stuck": stuck, **per_exit})
        print(f"run {run + 1:2d}: everyone out after {last_out:6.1f} s")

    # ---- summary ----
    print()
    print(f"average evacuation time : {statistics.mean(evacuation_times):.1f} s")
    print(f"spread (std. deviation) : {statistics.stdev(evacuation_times):.1f} s")
    print(f"fastest / slowest run   : {min(evacuation_times):.1f} s / {max(evacuation_times):.1f} s")
    under = sum(1 for t in evacuation_times if t <= TIME_LIMIT)
    print(f"runs within {TIME_LIMIT} s       : {under} of {NUMBER_OF_RUNS}")
    print("average people per exit :",
          ", ".join(f"{name} {exit_totals[name] / NUMBER_OF_RUNS:.0f}" for name in sim.EXIT_NAMES))

    # ---- files ----
    with open(os.path.join(OUTPUT_FOLDER, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw_map(hall)
    draw_people_left(all_out_times, total_people)
    draw_heatmap(hall, density_sum / NUMBER_OF_RUNS)
    print(f"\nfigures and results.csv saved in {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
