from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_three_policy_historical_efficiency import (
    DELAYED,
    GRID,
    INK,
    NO_DELAY,
    RANDOM,
    load_hourly,
    pad,
    ratio,
)


BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent.parent
RUNS = (
    (
        "GreenScore, 60-min delay",
        BASE / "timeseries" / "green_delay60_historical" / "completed_jobs.csv",
        DELAYED,
    ),
    (
        "GreenScore, no delay",
        BASE / "timeseries" / "green_nodelay_historical" / "completed_jobs.csv",
        NO_DELAY,
    ),
    (
        "Randomized, seed 42",
        BASE / "timeseries" / "random_seed42_historical" / "completed_jobs.csv",
        RANDOM,
    ),
)
OUTPUT_DIR = WORKSPACE / "DIRAC_simulation_reproduction_guide_figures"

SITE_ORDER = ("IN2P3-IRES", "SARA-MATRIX", "NCG-INGRID-PT", "TR-03-METU")
SITE_LABELS = {
    "IN2P3-IRES": "IN2P3-IRES (France)",
    "SARA-MATRIX": "SARA-MATRIX (Netherlands)",
    "NCG-INGRID-PT": "NCG-INGRID-PT (Portugal)",
    "TR-03-METU": "TR-03-METU (Turkey)",
}
SITE_COLORS = {
    "IN2P3-IRES": "#0072B2",
    "SARA-MATRIX": "#009E73",
    "NCG-INGRID-PT": "#E69F00",
    "TR-03-METU": "#D55E00",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / stem
    fig.savefig(output.with_suffix(".png"), dpi=320)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".svg"))
    print(output.with_suffix(".pdf"))


def plot_efficiency() -> None:
    loaded = [(label, load_hourly(path), color) for label, path, color in RUNS]
    job_counts = {int(data["jobs"]) for _, data, _ in loaded}
    workloads = [float(data["total_cpu"]) for _, data, _ in loaded]
    if len(job_counts) != 1 or not all(
        np.isclose(workloads[0], value, rtol=1e-10) for value in workloads[1:]
    ):
        raise ValueError("The three runs do not contain the same workload")

    size = max(len(np.asarray(data["cpu"])) for _, data, _ in loaded)
    for _, data, _ in loaded:
        for key in ("cpu", "carbon", "water"):
            data[key] = pad(np.asarray(data[key]), size)

    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    panels = (
        ("carbon", "Carbon efficiency", "Million normalized CPU-s / kgCO$_2$e"),
        ("water", "Water-scarcity efficiency", "Million normalized CPU-s / stress-L"),
    )
    hours = np.arange(size, dtype=float) + 0.5
    kernel = np.ones(3, dtype=float)

    for panel_index, (impact, title, ylabel) in enumerate(panels):
        ax = axes[panel_index]
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.65, alpha=0.8)
        for label, data, color in loaded:
            cpu = np.asarray(data["cpu"])
            impact_values = np.asarray(data[impact])
            hourly = ratio(cpu, impact_values) / 1_000_000.0
            rolling = ratio(
                np.convolve(cpu, kernel, mode="same"),
                np.convolve(impact_values, kernel, mode="same"),
            ) / 1_000_000.0
            ax.plot(
                hours,
                hourly,
                color=color,
                linewidth=0.75,
                marker="o",
                markersize=2.8,
                alpha=0.23,
            )
            ax.plot(hours, rolling, color=color, linewidth=2.1, label=label)

        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Elapsed simulation time (h)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, float(size))
        ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2", fontsize=8.0)
        ax.text(
            -0.11,
            1.04,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=INK,
        )

    fig.suptitle(
        "Portugal/Turkey sensitivity: environmental efficiency",
        y=0.98,
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.915,
        "Same 133,631-job workload; historical-interval carbon and water intensities",
        ha="center",
        fontsize=9.3,
        color="#53606b",
    )
    fig.text(
        0.5,
        0.02,
        "Faint points represent independent one-hour completion bins; bold curves are centered three-hour ratios. Higher is preferable.",
        ha="center",
        fontsize=8.5,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.81, bottom=0.17, wspace=0.28)
    save_figure(fig, "turkey_portugal_three_policy_carbon_water_efficiency")
    plt.close(fig)


def load_site_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["site"]] += 1
    return counts


def plot_allocation() -> None:
    configure_style()
    counts_by_policy = [(label, load_site_counts(path)) for label, path, _ in RUNS]
    totals = [sum(counts.values()) for _, counts in counts_by_policy]
    if len(set(totals)) != 1:
        raise ValueError("The three runs have different job counts")

    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    y_positions = np.arange(len(counts_by_policy))
    left = np.zeros(len(counts_by_policy), dtype=float)

    for site in SITE_ORDER:
        percentages = np.asarray(
            [100.0 * counts[site] / total for (_, counts), total in zip(counts_by_policy, totals)]
        )
        bars = ax.barh(
            y_positions,
            percentages,
            left=left,
            height=0.58,
            color=SITE_COLORS[site],
            label=SITE_LABELS[site],
        )
        for index, (bar, percentage) in enumerate(zip(bars, percentages)):
            if percentage >= 3.0:
                count = counts_by_policy[index][1][site]
                ax.text(
                    left[index] + percentage / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{percentage:.1f}%\n({count:,})",
                    ha="center",
                    va="center",
                    color="white" if site != "NCG-INGRID-PT" else INK,
                    fontsize=7.7,
                    fontweight="bold",
                )
        left += percentages

    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Share of completed jobs (%)")
    ax.set_yticks(y_positions, [label for label, _ in counts_by_policy])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(
        "Site allocation in the Portugal/Turkey sensitivity scenario",
        y=0.97,
        fontsize=14,
        weight="bold",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        frameon=True,
        edgecolor="#c8cdd2",
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.02,
        "Every policy completes the same 133,631 jobs; percentages report final placement, not site capacity.",
        ha="center",
        fontsize=8.5,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.21, right=0.985, top=0.70, bottom=0.16)
    save_figure(fig, "turkey_portugal_three_policy_site_allocation")
    plt.close(fig)


def main() -> None:
    plot_efficiency()
    plot_allocation()


if __name__ == "__main__":
    main()
