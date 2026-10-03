from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
PLOTS = BASE / "plots"
OUT = PLOTS / "history"


def read_site_history(path: Path) -> tuple[list[datetime], list[str], list[list[int]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        sites = [name for name in reader.fieldnames or [] if name != "timestamp"]
        rows = list(reader)

    times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    values = [[int(row[site]) for site in sites] for row in rows]
    return times, sites, values


def read_waiting_history(path: Path) -> tuple[list[datetime], list[int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    values = [int(row["waiting_jobs"]) for row in rows]
    return times, values


def columns(values: list[list[int]]) -> list[list[int]]:
    if not values:
        return []
    return [[row[i] for row in values] for i in range(len(values[0]))]


def total(values: list[list[int]]) -> list[int]:
    return [sum(row) for row in values]


def save_waiting_plot() -> None:
    times, waiting = read_waiting_history(PLOTS / "waiting_over_time.csv")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(times, waiting, where="post", color="tab:red")
    ax.set_title("Waiting Jobs Over Time")
    ax.set_xlabel("Simulated time")
    ax.set_ylabel("Waiting jobs")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "waiting_jobs.png", dpi=160)
    plt.close(fig)


def save_running_plot() -> None:
    times, sites, values = read_site_history(PLOTS / "jobs_running_per_site_over_time.csv")
    series = columns(values)

    fig, ax = plt.subplots(figsize=(14, 7))
    for site, site_values in zip(sites, series):
        ax.plot(times, site_values, linewidth=1.1, label=site)
    ax.set_title("Running Jobs Per Site Over Time")
    ax.set_xlabel("Simulated time")
    ax.set_ylabel("Running jobs")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "running_jobs_per_site.png", dpi=160)
    plt.close(fig)


def save_submitted_plot() -> None:
    times, sites, values = read_site_history(PLOTS / "jobs_submitted_per_site_over_time.csv")
    series = columns(values)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(times, series, labels=sites, alpha=0.85)
    ax.set_title("Jobs Submitted Per Site Over Time")
    ax.set_xlabel("Simulated time")
    ax.set_ylabel("Submitted jobs")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "submitted_jobs_per_site.png", dpi=160)
    plt.close(fig)


def save_total_running_vs_waiting_plot() -> None:
    waiting_times, waiting = read_waiting_history(PLOTS / "waiting_over_time.csv")
    running_times, _, running_values = read_site_history(PLOTS / "jobs_running_per_site_over_time.csv")
    total_running = total(running_values)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(running_times, total_running, color="tab:blue", label="Total running jobs")
    ax.plot(waiting_times, waiting, color="tab:red", label="Waiting jobs")
    ax.set_title("Total Running Jobs vs Waiting Jobs")
    ax.set_xlabel("Simulated time")
    ax.set_ylabel("Jobs")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "total_running_vs_waiting.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save_waiting_plot()
    save_running_plot()
    save_submitted_plot()
    save_total_running_vs_waiting_plot()
    print(f"Saved history plots to {OUT}")


if __name__ == "__main__":
    main()
