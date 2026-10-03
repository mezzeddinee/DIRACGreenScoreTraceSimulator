from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
TRACE_PATH = BASE / "trace6"
OUT_DIR = BASE / "plots"
PLOT_PATH = OUT_DIR / "trace6_trace_over_time.png"
CSV_PATH = OUT_DIR / "trace6_trace_over_time.csv"


def runtime_minutes(row: dict[str, str]) -> float:
    for names, divisor in (
        (("runtime_min", "runtime", "Runtime(min)"), 1.0),
        (("wallclocktime", "wallclock", "WallClockTime"), 60.0),
        (("norm_cpu_seconds", "cpu_seconds", "NormCPUTime(s)"), 60.0),
    ):
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return float(value) / divisor
    return 0.0


def load_trace() -> tuple[list[datetime], list[int], list[float], list[int]]:
    arrivals: dict[datetime, int] = defaultdict(int)
    runtime_sum: dict[datetime, float] = defaultdict(float)

    with TRACE_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            submit_time = datetime.fromisoformat(row["submit_time"].strip())
            minute = submit_time.replace(second=0, microsecond=0)
            arrivals[minute] += 1
            runtime_sum[minute] += runtime_minutes(row)

    minutes = sorted(arrivals)
    cumulative: list[int] = []
    total = 0
    for minute in minutes:
        total += arrivals[minute]
        cumulative.append(total)

    return (
        minutes,
        [arrivals[minute] for minute in minutes],
        [runtime_sum[minute] for minute in minutes],
        cumulative,
    )


def write_csv(
    minutes: list[datetime],
    arrivals: list[int],
    runtime_sum: list[float],
    cumulative: list[int],
) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "arrivals", "runtime_sum_min", "cumulative_jobs"])
        for row in zip(minutes, arrivals, runtime_sum, cumulative):
            minute, arrival_count, runtime_total, cumulative_jobs = row
            writer.writerow([minute.isoformat(), arrival_count, f"{runtime_total:.6f}", cumulative_jobs])


def save_plot(
    minutes: list[datetime],
    arrivals: list[int],
    runtime_sum: list[float],
    cumulative: list[int],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].step(minutes, arrivals, where="post", color="tab:purple", linewidth=1.2)
    axes[0].set_title("trace6 arrivals over time")
    axes[0].set_ylabel("Jobs/min")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(minutes, runtime_sum, where="post", color="tab:orange", linewidth=1.2)
    axes[1].set_title("trace6 runtime load over time")
    axes[1].set_ylabel("Runtime sum (min/min)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(minutes, cumulative, color="tab:blue", linewidth=1.6)
    axes[2].set_title("trace6 cumulative submitted jobs")
    axes[2].set_ylabel("Jobs")
    axes[2].set_xlabel("Submit time")
    axes[2].grid(True, alpha=0.25)

    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=170)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    minutes, arrivals, runtime_sum, cumulative = load_trace()
    write_csv(minutes, arrivals, runtime_sum, cumulative)
    save_plot(minutes, arrivals, runtime_sum, cumulative)
    print(f"Saved plot: {PLOT_PATH}")
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Minutes: {len(minutes)}")
    print(f"Jobs: {cumulative[-1] if cumulative else 0}")


if __name__ == "__main__":
    main()
