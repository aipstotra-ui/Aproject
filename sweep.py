"""
How much does it help when more people follow the ushers / Crowd Managers?

    python sweep.py

We run the concert emergency (case 4 data) with 60 %, 70 %, 80 %, 90 % and 100 % of people guided.
The people who are NOT guided are split like in the normal concert: 25 : 15 between habit and nearest.
Results: output/guided_sweep.png and output/guided_sweep.csv
"""
import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    # ---- figure: escape time against % guided ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = [100 * g for g, _, _, _ in summary]
    for g in GUIDED_SHARES:                                   # every run as a faint dot
        ys = [r["calculated_time_s"] for r in rows if r["guided"] == g]
        ax.scatter([100 * g] * len(ys), ys, color="#6250d6", alpha=0.18, s=14, linewidths=0)
    ax.errorbar(xs, [m for _, m, _, _ in summary], yerr=[s for _, _, s, _ in summary],
                color="#6250d6", marker="o", capsize=4, lw=2, label="average ± spread (30 runs)")
    ax.axhline(TIME_LIMIT, color="#d03b3b", ls="--", lw=1.2, label=f"{TIME_LIMIT} s limit")
    for x, (_, m, _, under) in zip(xs, summary):
        ax.annotate(f"{under}/{NUMBER_OF_RUNS}", (x, m), textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xticks(xs, [f"{x:.0f} %" for x in xs])
    ax.set_xlabel("people who follow the ushers / Crowd Managers")
    ax.set_ylabel("last person out (s, calculated)")
    ax.set_title("Concert emergency: the more people follow the plan, the faster")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(OUTPUT_FOLDER, "guided_sweep.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT_FOLDER}/guided_sweep.png and guided_sweep.csv")


if __name__ == "__main__":
    main()
