import csv
import math
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import simulation as sim

OUTPUT_FOLDER = "output"
SEED = 0
VOTE_RADIUS = 2.0
CASE = {"case": "LED", "data_case": 4, "name": "Concert, emergency, perfect guidance",
        "reaction": 10.0, "guided": 1.0, "habit": 0.0, "nearest": 0.0}
EXIT_COLOURS = {"FRONT": "#e8833a", "NW": "#1f77b4", "NE": "#17becf", "P1": "#2ca02c",
                "P2": "#98df8a", "P3": "#c5b0d5", "P4": "#9467bd"}


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    hall = sim.setup()
    grid = hall["grid"]
    data = sim.load_flow_data()
    rates = sim.rates_for_run(data, CASE["data_case"], SEED)
    people, _, _, _ = sim.run_once(SEED, hall, CASE, rates)
    last_out, _ = sim.calculated_escape_time(people, hall, rates)
    print(f"one run, 100 % guided: last person out after {last_out:.1f} s (calculated)")

    seated = [(p["start_square"], p["exit_used"]) for p in people if p["stair"] is None]

    exit_number = {name: i for i, name in enumerate(sim.EXIT_NAMES)}
    where = np.array([square for square, _ in seated])
    which = np.array([exit_number[e] for _, e in seated])
    section = np.full(grid.shape, -1)
    radius = VOTE_RADIUS / sim.CELL
    for row, col in zip(*np.nonzero(grid == sim.SEAT)):
        squared_distance = (where[:, 0] - row) ** 2 + (where[:, 1] - col) ** 2
        nearby = which[squared_distance <= radius ** 2]
        if len(nearby) == 0:
            nearby = [which[np.argmin(squared_distance)]]
        section[row, col] = Counter(nearby).most_common(1)[0][0]

    balcony = {}
    for p in people:
        if p["stair"] is not None:
            balcony.setdefault(p["stair_name"], Counter())[p["exit_used"]] += 1

    rows = []
    for area in ("orchestra", "mezzanine"):
        group = [e for (r, c), e in seated
                 if (math.hypot(*sim.square_centre(r, c)) < sim.ORCHESTRA_RADIUS) == (area == "orchestra")]
        counts = Counter(group)
        rows.append({"area": area, "people": len(group), **{e: counts.get(e, 0) for e in sim.EXIT_NAMES}})
    for name, counts in balcony.items():
        rows.append({"area": name, "people": sum(counts.values()), **{e: counts.get(e, 0) for e in sim.EXIT_NAMES}})
    with open(os.path.join(OUTPUT_FOLDER, "led_sections.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"  {row['area']:22s} " + ", ".join(f"{e} {row[e]}" for e in sim.EXIT_NAMES if row[e]))

    extent = [sim.X_MIN, sim.X_MAX, sim.Y_MIN, sim.Y_MAX]
    fig, ax = plt.subplots(figsize=(11, 8))
    background = np.where(grid == sim.WALL, 0, 1)
    ax.imshow(background, cmap=ListedColormap(["#3b3f45", "#f4f4f4"]), extent=extent, interpolation="nearest")
    colours = ListedColormap([EXIT_COLOURS[e] for e in sim.EXIT_NAMES])
    ax.imshow(np.ma.masked_where(section < 0, section), cmap=colours, vmin=-0.5, vmax=len(sim.EXIT_NAMES) - 0.5,
              extent=extent, interpolation="nearest")
    for name in sim.EXIT_NAMES:
        squares = [s for s, e in hall["exit_of_square"].items() if e == name]
        x = np.mean([sim.square_centre(*s)[0] for s in squares])
        y = np.mean([sim.square_centre(*s)[1] for s in squares])
        ax.scatter(x, y, s=260, color=EXIT_COLOURS[name], edgecolor="black", zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10,
                    fontweight="bold", color="black", zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.8))
    for name, how_many, spot in sim.BALCONY_STAIRS:
        counts = balcony["balcony " + name]
        main_exit, n = counts.most_common(1)[0]
        x, y = sim.square_centre(*sim.stair_landing(grid, spot))
        ax.scatter(x, y, s=140, marker="s", color=EXIT_COLOURS[main_exit], edgecolor="black", zorder=3)
        ax.annotate(f"balcony {name}\n{n}/{how_many} -> {main_exit}", (x, y), textcoords="offset points",
                    xytext=(0, -24), ha="center", fontsize=8, zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=[Patch(color=EXIT_COLOURS[e], label=e) for e in sim.EXIT_NAMES], title="LEDs point to",
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=7, frameon=False)
    ax.set_title("LED sections: every seat coloured by the exit the Crowd Managers' plan sends it to\n"
                 "(concert, 100 % guided, one run; each seat takes the exit most people within 2 m used)", loc="left", fontweight="bold")
    fig.savefig(os.path.join(OUTPUT_FOLDER, "led_map.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT_FOLDER}/led_map.png and led_sections.csv")


if __name__ == "__main__":
    main()
