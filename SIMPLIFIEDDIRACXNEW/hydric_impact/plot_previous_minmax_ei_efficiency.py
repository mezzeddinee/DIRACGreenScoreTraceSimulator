from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_multisite_ei_efficiency_over_time import (
    BASE,
    centered_ratio,
    load_hourly,
    pad,
    ratio,
)


GREEN = "#009E73"
ORANGE = "#E69F00"
GRID = "#d8dde1"
DEFAULT_GREEN = BASE / "timeseries/green_multisite/completed_jobs.csv"
DEFAULT_RANDOM = BASE / "timeseries/random_multisite/completed_jobs.csv"
DEFAULT_OUTPUT = BASE / "timeseries/minmax_previous_comparison/normcpu_per_minmax_ei_over_time"


def plot(green_path: Path, random_path: Path, output: Path) -> None:
    # With zero reference minima, Min--Max normalization equals maximum
    # normalization. K=1 selects the unbalanced EI definition.
    green = load_hourly(green_path, k_factor=1.0)
    random = load_hourly(random_path, k_factor=1.0)
    size = max(len(green["cpu"]), len(random["cpu"]))
    for dataset in (green, random):
        for key in ("cpu", "ei", "jobs"):
            dataset[key] = pad(dataset[key], size)
    hours = np.arange(size, dtype=float) + 0.5

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.8)
    for label, dataset, color in (
        ("Green-hydric", green, GREEN),
        ("Random", random, ORANGE),
    ):
        hourly = ratio(dataset["cpu"], dataset["ei"]) / 1_000_000.0
        rolling = centered_ratio(dataset["cpu"], dataset["ei"]) / 1_000_000.0
        ax.plot(hours, hourly, color=color, linewidth=0.8, marker="o",
                markersize=3.2, alpha=0.28)
        ax.plot(hours, rolling, color=color, linewidth=2.2,
                label=f"{label} (3-h window)")
    ax.set_title("Archived replay: Min--Max-normalized EI", loc="left", weight="bold")
    ax.set_xlabel("Elapsed simulation time (h)")
    ax.set_ylabel("Million normalized CPU-s / EI-weighted kWh")
    ax.set_xlim(0.0, float(size))
    ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2")
    fig.text(
        0.5,
        0.91,
        "CImin/max = 0/760 gCO$_2$e/kWh; HImin/max = 0/23.84 stress-L/kWh",
        ha="center",
        fontsize=9.5,
        color="#53606b",
    )
    fig.text(
        0.5,
        0.02,
        "Faint points are one-hour bins; bold curves are centered three-hour ratios. Higher is preferable.",
        ha="center",
        fontsize=8.8,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=320)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)

    green_eff = green["cpu"].sum() / green["ei"].sum() / 1_000_000.0
    random_eff = random["cpu"].sum() / random["ei"].sum() / 1_000_000.0
    print(f"green_eff={green_eff:.6f}")
    print(f"random_eff={random_eff:.6f}")
    print(f"efficiency_change={(green_eff / random_eff - 1.0) * 100.0:.4f}%")
    print(f"green_burden={green['ei'].sum():.9f}")
    print(f"random_burden={random['ei'].sum():.9f}")
    print(f"burden_reduction={(1.0-green['ei'].sum()/random['ei'].sum())*100.0:.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--green", type=Path, default=DEFAULT_GREEN)
    parser.add_argument("--random", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.green, args.random, args.output)
