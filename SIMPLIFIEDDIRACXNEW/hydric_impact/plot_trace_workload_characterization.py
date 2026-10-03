from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
TRACE_PATH = BASE.parent / "trace_2026_06_01.csv"
OUTPUT_STEM = BASE / "trace_workload_characterization"
MINUTE_PROFILE_PATH = BASE / "trace_minute_profile.csv"

TEAL = "#009E73"
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
INK = "#26333d"
GRID = "#d8dde1"


def load_trace() -> tuple[list[datetime], np.ndarray, np.ndarray, np.ndarray]:
    submitted: list[datetime] = []
    runtime: list[float] = []
    norm_cpu: list[float] = []
    cores: list[int] = []

    with TRACE_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            submitted.append(datetime.fromisoformat(row["submit_time"]))
            runtime.append(float(row["runtime_min"]))
            norm_cpu.append(float(row["norm_cpu_seconds"]))
            cores.append(int(row["cores_used"]))

    return submitted, np.asarray(runtime), np.asarray(norm_cpu), np.asarray(cores)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def build_minute_profile(
    submitted: list[datetime], runtime: np.ndarray, norm_cpu: np.ndarray
) -> tuple[list[datetime], np.ndarray, np.ndarray, np.ndarray]:
    day_start = min(submitted).replace(hour=0, minute=0, second=0, microsecond=0)
    minute_index = np.asarray([int((timestamp - day_start).total_seconds() // 60) for timestamp in submitted])
    minute_count = max(1440, int(minute_index.max()) + 1)
    arrivals = np.bincount(minute_index, minlength=minute_count)
    runtime_load = np.bincount(minute_index, weights=runtime, minlength=minute_count)
    cpu_load = np.bincount(minute_index, weights=norm_cpu, minlength=minute_count)
    timestamps = [day_start + timedelta(minutes=index) for index in range(minute_count)]
    return timestamps, arrivals, runtime_load, cpu_load


def write_minute_profile(
    timestamps: list[datetime], arrivals: np.ndarray, runtime_load: np.ndarray, cpu_load: np.ndarray
) -> None:
    with MINUTE_PROFILE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "job_arrivals",
                "submitted_runtime_min",
                "submitted_norm_cpu_seconds",
            ]
        )
        for timestamp, count, runtime_sum, cpu_sum in zip(timestamps, arrivals, runtime_load, cpu_load):
            writer.writerow([timestamp.isoformat(), int(count), f"{runtime_sum:.6f}", f"{cpu_sum:.6f}"])


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=INK,
        va="bottom",
    )


def main() -> None:
    submitted, runtime, norm_cpu, cores = load_trace()
    timestamps, arrivals, runtime_load, cpu_load = build_minute_profile(submitted, runtime, norm_cpu)
    write_minute_profile(timestamps, arrivals, runtime_load, cpu_load)

    percentiles = np.percentile(runtime, [50, 90, 95, 99])
    log_correlation = float(np.corrcoef(np.log10(runtime), np.log10(norm_cpu))[0, 1])
    duration_hours = (max(submitted) - min(submitted)).total_seconds() / 3600.0

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.65, alpha=0.75)

    # A: minute-level arrival process.
    ax = axes[0, 0]
    arrival_smooth = rolling_mean(arrivals.astype(float), 15)
    ax.fill_between(timestamps, arrivals, step="mid", color=TEAL, alpha=0.24, linewidth=0)
    ax.plot(timestamps, arrivals, color=TEAL, linewidth=0.45, alpha=0.65, label="Per minute")
    ax.plot(timestamps, arrival_smooth, color=INK, linewidth=1.7, label="15-min moving mean")
    ax.set_title("Arrival intensity over the day", loc="left", weight="bold")
    ax.set_ylabel("Submitted jobs / min")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper right", frameon=True, edgecolor="#c8cdd2")
    ax.text(
        0.02,
        0.93,
        f"mean {arrivals.mean():.1f}/min  ·  peak {arrivals.max()}/min",
        transform=ax.transAxes,
        color=INK,
        fontsize=9,
        va="top",
    )
    panel_label(ax, "A")

    # B: arriving work, expressed in raw runtime-minutes.
    ax = axes[0, 1]
    runtime_smooth = rolling_mean(runtime_load, 15)
    ax.fill_between(timestamps, runtime_load, step="mid", color=ORANGE, alpha=0.23, linewidth=0)
    ax.plot(timestamps, runtime_load, color=ORANGE, linewidth=0.45, alpha=0.7)
    ax.plot(timestamps, runtime_smooth, color=INK, linewidth=1.7, label="15-min moving mean")
    ax.set_yscale("log")
    ax.set_title("Submitted computational work", loc="left", weight="bold")
    ax.set_ylabel("Runtime-min arriving / min (log)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper right", frameon=True, edgecolor="#c8cdd2")
    ax.text(
        0.02,
        0.93,
        f"total {runtime.sum():,.0f} runtime-min",
        transform=ax.transAxes,
        color=INK,
        fontsize=9,
        va="top",
    )
    panel_label(ax, "B")

    # C: runtime complementary CDF exposes the long tail.
    ax = axes[1, 0]
    runtime_sorted = np.sort(runtime)
    ccdf = (runtime_sorted.size - np.arange(runtime_sorted.size)) / runtime_sorted.size
    ax.plot(runtime_sorted, ccdf, color=BLUE, linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Job-runtime tail", loc="left", weight="bold")
    ax.set_xlabel("Raw job runtime (min, log)")
    ax.set_ylabel("Fraction of jobs ≥ runtime")
    for percentile, value, color in zip([50, 95, 99], percentiles[[0, 2, 3]], [TEAL, ORANGE, VERMILLION]):
        ax.axvline(value, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        ax.text(
            value,
            0.72 if percentile == 50 else (0.12 if percentile == 95 else 0.026),
            f"p{percentile} {value:.2f} min",
            rotation=90,
            color=color,
            fontsize=8.5,
            ha="right",
            va="top",
        )
    panel_label(ax, "C")

    # D: the trace's CPU/runtime structure.
    ax = axes[1, 1]
    hexbin = ax.hexbin(
        runtime,
        norm_cpu,
        gridsize=58,
        xscale="log",
        yscale="log",
        bins="log",
        mincnt=1,
        cmap="viridis",
        linewidths=0.0,
    )
    colorbar = fig.colorbar(hexbin, ax=ax, pad=0.015, fraction=0.048)
    colorbar.set_label("Jobs per hexagon (log color)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    ax.set_title("Runtime–CPU workload fingerprint", loc="left", weight="bold")
    ax.set_xlabel("Raw job runtime (min, log)")
    ax.set_ylabel("Normalized CPU seconds (log)")
    ax.text(
        0.04,
        0.95,
        f"log–log Pearson r = {log_correlation:.3f}\nall jobs use one core",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=INK,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c8cdd2", "alpha": 0.94},
    )
    panel_label(ax, "D")

    fig.suptitle("Workload fingerprint of trace_2026_06_01", y=0.985, fontsize=17, weight="bold")
    fig.text(
        0.5,
        0.947,
        (
            f"{len(runtime):,} jobs · {duration_hours:.2f} h arrival window · "
            f"median runtime {percentiles[0]:.2f} min · p99 {percentiles[3]:.2f} min · "
            f"{int(np.unique(cores).size)} core-count class"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#53606b",
    )
    fig.text(
        0.5,
        0.016,
        "Runtime and CPU demand are trace inputs before site-specific performance scaling.",
        ha="center",
        fontsize=9,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.08, wspace=0.25, hspace=0.30)

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=320)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"))
    plt.close(fig)

    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(OUTPUT_STEM.with_suffix(".svg"))
    print(MINUTE_PROFILE_PATH)


if __name__ == "__main__":
    main()
