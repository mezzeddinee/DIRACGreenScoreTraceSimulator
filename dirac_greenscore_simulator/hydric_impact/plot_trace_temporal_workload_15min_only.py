from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from plot_trace_workload_characterization import (
    BLUE,
    GRID,
    INK,
    BASE,
    build_minute_profile,
    load_trace,
    panel_label,
    rolling_mean,
)


OUTPUT_STEM = BASE / "trace_temporal_workload_15min_only"


def main() -> None:
    submitted, runtime, norm_cpu, _ = load_trace()
    timestamps, arrivals, runtime_load, _ = build_minute_profile(
        submitted, runtime, norm_cpu
    )

    arrival_smooth = rolling_mean(arrivals.astype(float), 15)
    runtime_smooth = rolling_mean(runtime_load, 15)
    duration_hours = (max(submitted) - min(submitted)).total_seconds() / 3600.0
    average_load = float(runtime.sum()) / len(timestamps)

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.65, alpha=0.75)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax = axes[0]
    ax.plot(
        timestamps,
        arrival_smooth,
        color=BLUE,
        linewidth=1.7,
        label="15-min moving average",
    )
    ax.set_title("Arrival intensity over the day", loc="left", weight="bold")
    ax.set_xlabel("Submission time")
    ax.set_ylabel("Submitted jobs / min")
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

    ax = axes[1]
    ax.plot(
        timestamps,
        runtime_smooth,
        color=BLUE,
        linewidth=1.7,
        label="15-min moving average",
    )
    ax.set_title("Submitted computational work", loc="left", weight="bold")
    ax.set_xlabel("Submission time")
    ax.set_ylabel("Runtime-min arriving / min")
    ax.legend(loc="upper right", frameon=True, edgecolor="#c8cdd2")
    ax.text(
        0.02,
        0.93,
        f"total {runtime.sum():,.0f} runtime-min\nmean offered load {average_load:,.0f}",
        transform=ax.transAxes,
        color=INK,
        fontsize=9,
        va="top",
    )
    panel_label(ax, "B")

    fig.text(
        0.5,
        0.965,
        f"{len(runtime):,} single-core jobs · {duration_hours:.2f} h arrival window · 15-minute moving averages",
        ha="center",
        fontsize=9.5,
        color="#53606b",
    )
    fig.text(
        0.5,
        0.018,
        "Runtime demand is measured from trace inputs before site-specific performance scaling.",
        ha="center",
        fontsize=8.8,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.16, wspace=0.24)

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=320)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"))
    plt.close(fig)

    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(OUTPUT_STEM.with_suffix(".svg"))


if __name__ == "__main__":
    main()
