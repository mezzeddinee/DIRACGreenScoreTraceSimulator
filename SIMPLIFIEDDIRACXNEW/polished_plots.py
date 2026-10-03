from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
PLOTS = BASE / "plots"
OUT = PLOTS / "polished"

BG = "#090d18"
PANEL = "#111827"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
CYAN = "#22d3ee"
GREEN = "#34d399"
PINK = "#fb7185"
PURPLE = "#a78bfa"
GRID = "#334155"


def load_series(path: Path) -> tuple[list[datetime], list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = [name for name in reader.fieldnames or [] if name != "timestamp"]
        rows = list(reader)
    times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    values = np.array([[int(row[name]) for name in names] for row in rows], dtype=float)
    return times, names, values


def style_axis(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, alpha=0.25, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_title(fig, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.965, title, color=TEXT, fontsize=24, weight="bold", va="top")
    fig.text(0.055, 0.925, subtitle, color=MUTED, fontsize=10, va="top")


def add_kpi(fig, x: float, label: str, value: str, accent: str) -> None:
    fig.text(x, 0.86, value, color=accent, fontsize=20, weight="bold")
    fig.text(x, 0.825, label.upper(), color=MUTED, fontsize=8, weight="bold")


def short_site_name(name: str) -> str:
    return name.removeprefix("UKI-").removeprefix("NORTHGRID-").removeprefix("SOUTHGRID-")


def save_dashboard(
    times: list[datetime],
    names: list[str],
    running: np.ndarray,
    waiting: np.ndarray,
    submitted: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    grid = fig.add_gridspec(2, 2, left=0.055, right=0.97, bottom=0.07, top=0.76, hspace=0.32, wspace=0.22)
    add_title(fig, "SIMULATION PULSE", "40,040 jobs · random scheduling · 12 sites · 1-minute ticks")
    add_kpi(fig, 0.055, "Completed jobs", "40,040", CYAN)
    add_kpi(fig, 0.255, "Total carbon", "10.51 kgCO₂", GREEN)
    add_kpi(fig, 0.48, "Median turnaround", "5.65 min", PURPLE)
    add_kpi(fig, 0.71, "P99 queue wait", "0.98 min", PINK)

    total_running = running.sum(axis=1)
    ax = fig.add_subplot(grid[0, :])
    style_axis(ax)
    ax.fill_between(times, total_running, color=CYAN, alpha=0.18)
    ax.plot(times, total_running, color=CYAN, linewidth=1.5, label="Running")
    ax.plot(times, waiting[:, 0], color=PINK, linewidth=1.2, label="Waiting")
    ax.set_title("System load over simulated time", color=TEXT, loc="left", weight="bold")
    ax.set_ylabel("Jobs", color=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(frameon=False, labelcolor=TEXT, ncol=2, loc="upper right")

    totals = submitted.sum(axis=0)
    order = np.argsort(totals)
    ax = fig.add_subplot(grid[1, 0])
    style_axis(ax)
    ax.barh([short_site_name(names[i]) for i in order], totals[order], color=PURPLE, alpha=0.9)
    ax.set_title("Jobs placed per site", color=TEXT, loc="left", weight="bold")
    ax.set_xlabel("Jobs", color=MUTED)
    ax.grid(axis="y", visible=False)

    peak = running.max(axis=0)
    avg = running.mean(axis=0)
    order = np.argsort(peak)
    ax = fig.add_subplot(grid[1, 1])
    style_axis(ax)
    y = np.arange(len(names))
    ax.hlines(y, avg[order], peak[order], color=GRID, linewidth=3)
    ax.scatter(avg[order], y, color=GREEN, s=35, label="Average")
    ax.scatter(peak[order], y, color=PINK, s=35, label="Peak")
    ax.set_yticks(y, [names[i] for i in order])
    ax.set_title("Average vs peak concurrent jobs", color=TEXT, loc="left", weight="bold")
    ax.set_xlabel("Concurrent jobs", color=MUTED)
    ax.legend(frameon=False, labelcolor=TEXT, ncol=2, loc="lower right")

    fig.savefig(OUT / "simulation_dashboard.png", dpi=180, facecolor=BG)
    plt.close(fig)


def save_site_ranking(names: list[str], submitted: np.ndarray) -> None:
    totals = submitted.sum(axis=0)
    order = np.argsort(totals)
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    style_axis(ax)
    colors = [PURPLE] * len(order)
    colors[-1] = CYAN
    bars = ax.barh([names[i] for i in order], totals[order], color=colors)
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in totals[order]], padding=5, color=TEXT, fontsize=9)
    ax.set_xlim(0, totals.max() * 1.18)
    ax.set_title("WHERE THE JOBS WENT", color=TEXT, loc="left", fontsize=20, weight="bold", pad=20)
    ax.text(0, 1.01, "Random scheduling run · total assigned jobs by site", transform=ax.transAxes, color=MUTED)
    ax.set_xlabel("Jobs assigned", color=MUTED)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "site_throughput_ranking.png", dpi=180, facecolor=BG)
    plt.close(fig)


def save_heatmap(times: list[datetime], names: list[str], running: np.ndarray) -> None:
    # Aggregate minute-level snapshots into 15-minute averages for a readable heatmap.
    block = 15
    usable = len(running) - (len(running) % block)
    heat = running[:usable].reshape(-1, block, len(names)).mean(axis=1).T
    sample_times = times[:usable:block]

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    image = ax.imshow(heat, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(np.arange(len(names)), names, color=TEXT, fontsize=9)
    positions = np.linspace(0, len(sample_times) - 1, 8, dtype=int)
    ax.set_xticks(positions, [sample_times[i].strftime("%d %b\n%H:%M") for i in positions], color=MUTED)
    ax.set_title("SITE ACTIVITY HEATMAP", color=TEXT, loc="left", fontsize=20, weight="bold", pad=20)
    ax.text(0, 1.01, "Average concurrent jobs per 15-minute window", transform=ax.transAxes, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Concurrent jobs", color=MUTED)
    colorbar.ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "site_activity_heatmap.png", dpi=180, facecolor=BG)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    times, names, running = load_series(PLOTS / "jobs_running_per_site_over_time.csv")
    _, submitted_names, submitted = load_series(PLOTS / "jobs_submitted_per_site_over_time.csv")
    _, waiting_names, waiting = load_series(PLOTS / "waiting_over_time.csv")
    if names != submitted_names or waiting_names != ["waiting_jobs"]:
        raise ValueError("Unexpected telemetry CSV columns")
    save_dashboard(times, names, running, waiting, submitted)
    save_site_ranking(names, submitted)
    save_heatmap(times, names, running)
    print(f"Saved polished plots to {OUT}")


if __name__ == "__main__":
    main()
