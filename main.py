"""
Run the Bailey Hall simulation for the two emergency cases, many times each, and make the figures.

    python main.py

The main result is the CALCULATED time for the last person to get out (simulation.calculated_escape_time):
people per doorway (counted in the simulation) / the doorway's flow rate, chained from the seats to the exits.

Results go into the folder output/:
    map.png           the hall as the computer sees it (doorways with data marked)
    escape_time.png   calculated time for the last person to get out, every run, per case
    doorway_times.png when each doorway lets its last person through (which doorway decides the time)
    heatmap.png      how crowded each square was, one panel per case
    flow_check.png   busiest flow through the stairs / mezzanine doors / foyer doors: simulation vs our data
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
from matplotlib.patches import Patch

import simulation as sim

NUMBER_OF_RUNS = 30        # how many times we repeat each case
TIME_LIMIT = 180           # the fire code says everyone must be out within 180 s
OUTPUT_FOLDER = "output"
CASE_COLOURS = ["#1baf7a", "#6250d6"]   # case 3 (students), case 4 (concert)
PLACES = [("stair", "stairs", "Stairs (orchestra -> mezzanine)"),
          ("mezz door", "corridor", "Mezzanine doors (corridor)"),
          ("foyer door", "foyer", "Foyer doors (all 3)")]


# ------------------------------------------------------------------------------- measuring flow
BUSIEST_WINDOW = 20.0      # seconds: we look for the busiest 20 s at each doorway


def busiest_flow(times):
    """The most people that went through in any 20-second stretch, divided by 20 (people per second)."""
    times = sorted(times)
    most = 0
    start = 0
    for end in range(len(times)):
        while times[end] - times[start] >= BUSIEST_WINDOW:
            start += 1                                     # slide the window forward
        most = max(most, end - start + 1)
    return most / BUSIEST_WINDOW


def simulated_flows(gate_times):
    """For each kind of doorway in one run:
      average = people through / (time of the last - time of the first)   (includes quiet moments)
      busiest = flow in the busiest 20 s                                    (compare with our data)
    Stairs and mezzanine doors: averaged over the doorways at least 10 people used.
    Foyer doors: all 3 doors together, like in the data."""
    flows = {}
    for gate_kind, _, _ in PLACES:
        lists = [times for gate, times in gate_times.items() if gate.startswith(gate_kind)]
        if gate_kind == "foyer door":
            lists = [sum(lists, [])]                       # put the 3 foyer doors together
        lists = [times for times in lists if len(times) >= 10]
        if not lists:
            flows[gate_kind] = {"average": 0.0, "busiest": 0.0}
            continue
        averages = [len(t) / (max(t) - min(t) + sim.TIME_STEP) for t in lists]
        busiest = [busiest_flow(t) for t in lists]
        flows[gate_kind] = {"average": statistics.mean(averages), "busiest": statistics.mean(busiest)}
    return flows


# ------------------------------------------------------------------------------- figures
def draw_map(hall):
    """Picture of the grid, with the doorways that use our data in orange."""
    grid = hall["grid"].copy()
    for square in hall["gate_of_square"]:
        grid[square] = 4                                   # colour doorways with data differently
    colours = ListedColormap(["#3b3f45", "#f2efe8", "#c9d9ec", "#2e9e5b", "#e8833a"])
    extent = [sim.X_MIN, sim.X_MAX, sim.Y_MIN, sim.Y_MAX]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.imshow(grid, cmap=colours, vmin=0, vmax=4, extent=extent, interpolation="nearest")
    for name in sim.EXIT_NAMES:
        squares = [square for square, exit_name in hall["exit_of_square"].items() if exit_name == name]
        xs = [sim.square_centre(r, c)[0] for r, c in squares]
        ys = [sim.square_centre(r, c)[1] for r, c in squares]
        label_y = np.mean(ys) + 1.5 if name == "FRONT" else np.mean(ys) - 1.3
        ax.text(np.mean(xs), label_y, name, ha="center", va="top", fontsize=9, fontweight="bold", color="#1d6b3d")
    for _, _, where in sim.BALCONY_STAIRS:
        x, y = sim.square_centre(*sim.stair_landing(hall["grid"], where))
        ax.plot(x, y, marker="v", color="#8e44ad", markersize=9)
    ax.plot([], [], marker="v", color="#8e44ad", linestyle="none", label="balcony stairs arrive")
    for value, label in [(1, "floor"), (2, "seats"), (4, "doorways with our data"), (3, "exits"), (0, "wall")]:
        ax.plot([], [], marker="s", linestyle="none", markersize=10, color=colours(value), label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    ax.set_title("Bailey Hall as a grid of 0.5 m x 0.5 m squares", loc="left", fontweight="bold")
    ax.set_xlabel("metres (west - east)")
    ax.set_ylabel("metres (foyer - stage)")
    fig.savefig(os.path.join(OUTPUT_FOLDER, "map.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_escape_times(results):
    """One dot per run: the calculated time for the last person to get out, per case."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for row, (case, colour) in enumerate(zip(sim.CASES, CASE_COLOURS)):
        times = results[case["case"]]["times"]
        wobble = np.linspace(-0.12, 0.12, len(times))         # spread the dots a little so they don't overlap
        ax.scatter(times, row + wobble, color=colour, s=22, alpha=0.8)
        mean_time = statistics.mean(times)
        ax.plot([mean_time, mean_time], [row - 0.28, row + 0.28], color="#222222", linewidth=2)
        ax.text(mean_time, row + 0.33, f"average {mean_time:.0f} s", ha="center", fontsize=9)
    ax.axvline(TIME_LIMIT, color="#c0392b", linestyle="--", linewidth=1.2)
    ax.text(TIME_LIMIT + 2, -0.45, "180 s limit", color="#c0392b", fontsize=9)
    ax.set_yticks(range(len(sim.CASES)), [f"Case {c['case']}: {c['name']}" for c in sim.CASES])
    ax.set_ylim(-0.6, len(sim.CASES) - 0.4)
    ax.set_xlabel("seconds until the last person is out (calculated)")
    ax.set_title(f"Time for the last person to get out ({NUMBER_OF_RUNS} runs per case, one dot per run)",
                 loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_FOLDER, "escape_time.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_doorway_times(results):
    """For each doorway: average time when its last person goes through. The longest bar decides."""
    kinds = [("stair", "#9db7d5", "stairs"), ("stage", "#9db7d5", None), ("balcony", "#b8a9e0", "balcony stairs"),
             ("mezz door", "#1f4e8c", "mezzanine doors"), ("foyer door", "#e8833a", "foyer doors"),
             ("", "#2e9e5b", "exits")]

    def colour_of(doorway):
        for start, colour, _ in kinds:
            if doorway.startswith(start):
                return colour
        return "#2e9e5b"

    fig, axes = plt.subplots(1, len(sim.CASES), figsize=(13, 7.5), sharex=True)
    for ax, case in zip(axes, sim.CASES):
        finishes = results[case["case"]]["doorway_finish"]
        names = sorted(finishes, key=lambda d: (sim.doorway_level(d), d))
        averages = [statistics.mean(finishes[d]) for d in names]
        people = [statistics.mean(results[case["case"]]["doorway_people"][d]) for d in names]
        ax.barh(range(len(names)), averages, color=[colour_of(d) for d in names])
        for i, (avg, n) in enumerate(zip(averages, people)):
            ax.text(avg + 2, i, f"{n:.0f} people", va="center", fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(TIME_LIMIT, color="#c0392b", linestyle="--", linewidth=1)
        ax.set_title(f"Case {case['case']}: {case['name']}", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("seconds until the doorway's last person is through")
        ax.grid(axis="x", alpha=0.3)
    legend_boxes = [Patch(color=colour, label=label) for start, colour, label in kinds if label]
    fig.legend(handles=legend_boxes, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("When each doorway lets its last person through (average over runs)", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(OUTPUT_FOLDER, "doorway_times.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_heatmaps(hall, results):
    """One crowding map per case, all on the same colour scale."""
    grid = hall["grid"]
    extent = [sim.X_MIN, sim.X_MAX, sim.Y_MIN, sim.Y_MAX]
    top = max(float(results[c["case"]]["density"].max()) for c in sim.CASES)
    fig, axes = plt.subplots(1, len(sim.CASES), figsize=(14, 4.6))
    for ax, case in zip(axes.flat, sim.CASES):
        density = np.ma.masked_where(grid == sim.WALL, results[case["case"]]["density"])
        ax.imshow(grid == sim.WALL, cmap=ListedColormap(["white", "#3b3f45"]), extent=extent, interpolation="nearest")
        picture = ax.imshow(density, cmap="YlOrRd", vmin=0, vmax=top, extent=extent, interpolation="nearest")
        ax.set_title(f"Case {case['case']}: {case['name']}", loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(top=0.86, right=0.9, wspace=0.05)
    colour_bar_space = fig.add_axes([0.92, 0.15, 0.012, 0.65])      # its own strip on the right
    bar = fig.colorbar(picture, cax=colour_bar_space)
    bar.set_label("share of time the square was occupied")
    fig.suptitle("Where people crowd (average over all runs)", x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(OUTPUT_FOLDER, "heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_flow_check(results, data):
    """Bars: measured flow in our data vs the busiest 20 s and the average in this simulation."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(sim.CASES))
    for ax, (gate_kind, data_place, title) in zip(axes, PLACES):
        measured = [statistics.mean(float(row["Flow_people_per_s"]) for row in data[data_place][c["case"]])
                    for c in sim.CASES]
        busiest = [statistics.mean(f["busiest"] for f in results[c["case"]]["flows"][gate_kind]) for c in sim.CASES]
        average = [statistics.mean(f["average"] for f in results[c["case"]]["flows"][gate_kind]) for c in sim.CASES]
        ax.bar(x - 0.27, measured, width=0.26, color="#9db7d5", label="our data")
        ax.bar(x, busiest, width=0.26, color="#1f4e8c", label="simulation, busiest 20 s")
        ax.bar(x + 0.27, average, width=0.26, color="#e8833a", label="simulation, average")
        ax.set_xticks(x, [f"Case {c['case']}" for c in sim.CASES])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("people per second")
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(os.path.join(OUTPUT_FOLDER, "flow_check.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------- main program
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    hall = sim.setup()
    data = sim.load_flow_data()
    total_people = sim.ORCHESTRA_PEOPLE + sim.MEZZANINE_PEOPLE + sum(n for _, n, _ in sim.BALCONY_STAIRS)

    results = {}
    rows = []
    for case in sim.CASES:
        number = case["case"]
        results[number] = {"times": [], "simulated_times": [], "density": np.zeros(hall["grid"].shape),
                           "doorway_finish": {}, "doorway_people": {},
                           "exits": {name: 0 for name in sim.EXIT_NAMES},
                           "flows": {kind: [] for kind, _, _ in PLACES}}
        print(f"\nCase {number}: {case['name']}")
        for run in range(NUMBER_OF_RUNS):
            rates = sim.rates_for_run(data, number, run)          # this run's measured flow rates
            people, times_occupied, steps, gate_times = sim.run_once(run, hall, case, rates)
            out_times = [p["out_time"] for p in people if p["out_time"] is not None]
            stuck = total_people - len(out_times)
            simulated_last_out = max(out_times) if stuck == 0 else float("inf")
            # the main result: calculated from people per doorway / flow rate, chained to the exits
            last_out, doorways = sim.calculated_escape_time(people, hall, rates)

            r = results[number]
            r["times"].append(last_out)
            r["simulated_times"].append(simulated_last_out)
            for doorway, info in doorways.items():
                r["doorway_finish"].setdefault(doorway, []).append(info["finish"])
                r["doorway_people"].setdefault(doorway, []).append(info["people"])
            r["density"] += times_occupied / steps / NUMBER_OF_RUNS
            per_exit = {name: 0 for name in sim.EXIT_NAMES}
            for p in people:
                if p["exit_used"] is not None:
                    per_exit[p["exit_used"]] += 1
                    r["exits"][p["exit_used"]] += 1
            flows = simulated_flows(gate_times)
            for kind in flows:
                r["flows"][kind].append(flows[kind])
            slowest = max(doorways, key=lambda d: doorways[d]["finish"] if sim.doorway_level(d) < 4 else -1)
            rows.append({"case": number, "case_name": case["name"], "run": run,
                         "calculated_time_s": round(last_out, 1), "simulated_time_s": round(simulated_last_out, 1),
                         "slowest_inner_doorway": slowest, "people_stuck": stuck, **per_exit,
                         **{f"busiest_flow_{kind.replace(' ', '_')}": round(flows[kind]["busiest"], 3) for kind in flows}})
            print(f"  run {run + 1:2d}: last person out after {last_out:6.1f} s (calculated)")

    # ---- summary ----
    print("\n================ summary ================")
    for case in sim.CASES:
        r = results[case["case"]]
        times = r["times"]
        under = sum(1 for t in times if t <= TIME_LIMIT)
        print(f"\nCase {case['case']}: {case['name']}")
        print(f"  last person out : {statistics.mean(times):.1f} s on average (spread {statistics.stdev(times):.1f} s, "
              f"fastest {min(times):.1f}, slowest {max(times):.1f})  [calculated]")
        print(f"  (the step-by-step simulation itself took {statistics.mean(r['simulated_times']):.1f} s on average)")
        finishes = {d: statistics.mean(v) for d, v in r["doorway_finish"].items()}
        last_three = sorted(finishes, key=finishes.get, reverse=True)[:3]
        print("  doorways that finish last: " + ", ".join(f"{d} {finishes[d]:.0f} s" for d in last_three))
        print(f"  within {TIME_LIMIT} s     : {under} of {NUMBER_OF_RUNS} runs")
        print("  people per exit : " + ", ".join(f"{n} {r['exits'][n] / NUMBER_OF_RUNS:.0f}" for n in sim.EXIT_NAMES))
        for kind, place, title in PLACES:
            measured = statistics.mean(float(row["Flow_people_per_s"]) for row in data[place][case["case"]])
            busiest = statistics.mean(f["busiest"] for f in r["flows"][kind])
            average = statistics.mean(f["average"] for f in r["flows"][kind])
            print(f"  {title:34s}: data {measured:.2f} p/s | simulation busiest 20 s {busiest:.2f}, average {average:.2f} p/s")

    # ---- files ----
    with open(os.path.join(OUTPUT_FOLDER, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw_map(hall)
    draw_escape_times(results)
    draw_doorway_times(results)
    draw_heatmaps(hall, results)
    draw_flow_check(results, data)
    print(f"\nfigures and results.csv saved in {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
