import csv
import os
import statistics

import numpy as np

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
    ("A: between mezz doors 3|4 and 6|7", ["mezz door 3 | 4", "mezz door 6 | 7"]),
    ("B: between mezz doors 4|5 and 5|6", ["mezz door 4 | 5", "mezz door 5 | 6"]),
]


def no_barrier_times():
    with open(os.path.join(OUTPUT_FOLDER, "results.csv")) as f:
        return [float(row["calculated_time_s"]) for row in csv.DictReader(f) if row["case"] == CASE["case"]]


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
    baseline = no_barrier_times()
    print("\n================ summary ================")
    print(f"\nNo barriers (from results.csv): {statistics.mean(baseline):.1f} s +- {statistics.stdev(baseline):.1f},"
          f" within {TIME_LIMIT} s in {sum(1 for x in baseline if x <= TIME_LIMIT)}/{len(baseline)}")
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

    print(f"\nsaved {OUTPUT_FOLDER}/barriers.csv")


if __name__ == "__main__":
    main()
