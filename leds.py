import csv
import math
import os
from collections import Counter

import numpy as np

import simulation as sim

OUTPUT_FOLDER = "output"
SEED = 0
VOTE_RADIUS = 2.0          # metres: each seat takes the exit most people within this distance used
CASE = {"case": "LED", "data_case": 4, "name": "Concert, emergency, perfect guidance",
        "reaction": 10.0, "guided": 1.0, "habit": 0.0, "nearest": 0.0}


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    hall = sim.setup()
    grid = hall["grid"]
    data = sim.load_flow_data()
    rates = sim.rates_for_run(data, CASE["data_case"], SEED)
    people, _, _, _ = sim.run_once(SEED, hall, CASE, rates)
    last_out, _ = sim.calculated_escape_time(people, hall, rates)
    print(f"one run, 100 % guided: last person out after {last_out:.1f} s (calculated)")

    # 1. every person who started in a seat: where they sat and which exit they used
    seated = [(p["start_square"], p["exit_used"]) for p in people if p["stair"] is None]

    # 2. colour every seat square by the exit MOST people within 1.5 m of it used (a "majority vote").
    #    This gives clean sections, like real LED strips would need, and empty seats get a colour too.
    exit_number = {name: i for i, name in enumerate(sim.EXIT_NAMES)}
    where = np.array([square for square, _ in seated])
    which = np.array([exit_number[e] for _, e in seated])
    section = np.full(grid.shape, -1)
    radius = VOTE_RADIUS / sim.CELL                       # in squares
    for row, col in zip(*np.nonzero(grid == sim.SEAT)):
        squared_distance = (where[:, 0] - row) ** 2 + (where[:, 1] - col) ** 2
        nearby = which[squared_distance <= radius ** 2]
        if len(nearby) == 0:
            nearby = [which[np.argmin(squared_distance)]]   # nobody close: copy the closest person
        section[row, col] = Counter(nearby).most_common(1)[0][0]

    # 3. balcony staircases: which exits their people used
    balcony = {}
    for p in people:
        if p["stair"] is not None:
            balcony.setdefault(p["stair_name"], Counter())[p["exit_used"]] += 1

    # ---- table: share of each seating area going to each exit ----
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

    print(f"saved {OUTPUT_FOLDER}/led_sections.csv")


if __name__ == "__main__":
    main()
