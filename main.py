import csv
import os
import statistics

import numpy as np

import simulation as sim

NUMBER_OF_RUNS = 30        # how many times we repeat each case
TIME_LIMIT = 180           # the fire code says everyone must be out within 180 s
OUTPUT_FOLDER = "output"
PLACES = [("stair", "stairs", "Stairs (orchestra -> mezzanine)"),
          ("mezz door", "corridor", "Mezzanine doors (corridor)"),
          ("foyer door", "foyer", "Foyer doors (all 3)")]


# ------------------------------------------------------------------------------- measuring flow
BUSIEST_WINDOW = 20.0      # seconds: we look for the busiest 20 s at each doorway


def busiest_flow(times):
    times = sorted(times)
    most = 0
    start = 0
    for end in range(len(times)):
        while times[end] - times[start] >= BUSIEST_WINDOW:
            start += 1                                     # slide the window forward
        most = max(most, end - start + 1)
    return most / BUSIEST_WINDOW


def simulated_flows(gate_times):
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
            rates = sim.rates_for_run(data, case["data_case"], run)   # this run's measured flow rates
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
            measured = statistics.mean(float(row["Flow_people_per_s"]) for row in data[place][case["data_case"]])
            busiest = statistics.mean(f["busiest"] for f in r["flows"][kind])
            average = statistics.mean(f["average"] for f in r["flows"][kind])
            print(f"  {title:34s}: data {measured:.2f} p/s | simulation busiest 20 s {busiest:.2f}, average {average:.2f} p/s")

    # ---- files ----
    with open(os.path.join(OUTPUT_FOLDER, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nresults.csv saved in {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
