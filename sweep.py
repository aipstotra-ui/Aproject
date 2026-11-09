import csv
import os
import statistics

import simulation as sim

NUMBER_OF_RUNS = 30
TIME_LIMIT = 180
OUTPUT_FOLDER = "output"
GUIDED_SHARES = [0.60, 0.70, 0.80, 0.90, 1.00]


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    hall = sim.setup()
    data = sim.load_flow_data()

    rows = []
    summary = []
    for guided in GUIDED_SHARES:
        rest = 1.0 - guided                                   # people who don't follow the ushers
        case = {"case": f"{guided:.0%} guided", "data_case": 4, "name": "Concert, emergency",
                "reaction": 10.0, "guided": guided, "habit": rest * 25 / 40, "nearest": rest * 15 / 40}
        times = []
        for run in range(NUMBER_OF_RUNS):
            rates = sim.rates_for_run(data, 4, run)
            people, _, _, _ = sim.run_once(run, hall, case, rates)
            last_out, _ = sim.calculated_escape_time(people, hall, rates)
            times.append(last_out)
            rows.append({"guided": guided, "run": run, "calculated_time_s": round(last_out, 1)})
        under = sum(1 for t in times if t <= TIME_LIMIT)
        summary.append((guided, statistics.mean(times), statistics.stdev(times), under))
        print(f"{guided:4.0%} guided: {statistics.mean(times):6.1f} s +- {statistics.stdev(times):4.1f}, "
              f"within {TIME_LIMIT} s in {under}/{NUMBER_OF_RUNS} runs")

    with open(os.path.join(OUTPUT_FOLDER, "guided_sweep.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {OUTPUT_FOLDER}/guided_sweep.csv")


if __name__ == "__main__":
    main()
