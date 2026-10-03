from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_GREEN = BASE / "timeseries" / "green_multisite" / "completed_jobs.csv"
DEFAULT_RANDOM = BASE / "timeseries" / "random_multisite" / "completed_jobs.csv"
DEFAULT_OUTPUT = BASE / "multisite_interval_normcpu_efficiency_over_time"

GREEN = "#009E73"
ORANGE = "#E69F00"
INK = "#26333d"
GRID = "#d8dde1"


def load_hourly(path: Path) -> dict[str, np.ndarray]:
    rows: list[tuple[datetime, datetime, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    datetime.fromisoformat(row["submit_time"]),
                    datetime.fromisoformat(row["finish_time"]),
                    float(row["norm_cpu_seconds"]),
                    float(row["carbon_kg"]),
                    float(row["water_impact_stress_l"]),
                )
            )
    if not rows:
        raise ValueError(f"No completed jobs in {path}")

    origin = min(row[0] for row in rows)
    hour_index = np.asarray(
        [max(0, int((row[1] - origin).total_seconds() // 3600)) for row in rows],
        dtype=int,
    )
    size = int(hour_index.max()) + 1
    cpu = np.bincount(hour_index, weights=[row[2] for row in rows], minlength=size)
    carbon = np.bincount(hour_index, weights=[row[3] for row in rows], minlength=size)
    water = np.bincount(hour_index, weights=[row[4] for row in rows], minlength=size)
    jobs = np.bincount(hour_index, minlength=size)
    return {"cpu": cpu, "carbon": carbon, "water": water, "jobs": jobs}


def padded(values: np.ndarray, size: int) -> np.ndarray:
    return np.pad(values, (0, size - values.size), constant_values=0.0)


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def centered_window_ratio(
    numerator: np.ndarray, denominator: np.ndarray, window: int = 3
) -> np.ndarray:
    if numerator.ndim != 1 or denominator.ndim != 1:
        raise ValueError("Centered-window inputs must be one-dimensional")
    if numerator.shape != denominator.shape:
        raise ValueError("Centered-window inputs must have matching shapes")
    if window < 1:
        raise ValueError("Centered-window size must be positive")
    if numerator.size == 0:
        return np.asarray([], dtype=float)

    kernel = np.ones(window, dtype=float)
    start = (window - 1) // 2

    def centered_sum(values: np.ndarray) -> np.ndarray:
        full = np.convolve(values, kernel, mode="full")
        return full[start : start + values.size]

    return ratio(
        centered_sum(numerator),
        centered_sum(denominator),
    )


def plot(
    green_path: Path,
    random_path: Path,
    output_stem: Path,
    comparison_label: str | None = None,
    water_inset: bool = False,
) -> None:
    green = load_hourly(green_path)
    random = load_hourly(random_path)
    size = max(green["cpu"].size, random["cpu"].size)
    for dataset in (green, random):
        for key in ("cpu", "carbon", "water", "jobs"):
            dataset[key] = padded(dataset[key], size)

    hours = np.arange(size, dtype=float) + 0.5

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [
        ("carbon", "Hourly carbon efficiency", "Million normalized CPU-s / kgCO$_2$e"),
        ("water", "Hourly water-scarcity efficiency", "Million normalized CPU-s / stress-L"),
    ]

    for panel_index, (impact_key, title, ylabel) in enumerate(panels):
        ax = axes[panel_index]
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.65, alpha=0.8)

        for label, dataset, color in (
            ("GreenScore-based", green, GREEN),
            ("Random", random, ORANGE),
        ):
            hourly = ratio(dataset["cpu"], dataset[impact_key]) / 1_000_000.0
            rolling = centered_window_ratio(dataset["cpu"], dataset[impact_key], window=3) / 1_000_000.0
            ax.plot(
                hours,
                hourly,
                color=color,
                linewidth=0.8,
                marker="o",
                markersize=3.2,
                alpha=0.28,
            )
            ax.plot(hours, rolling, color=color, linewidth=2.2, label=f"{label} (3-h window)")

        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Elapsed simulation time (h)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, float(size))
        ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2")

        if water_inset and impact_key == "water":
            inset = ax.inset_axes([0.07, 0.08, 0.43, 0.34])
            random_hourly = ratio(random["cpu"], random["water"]) / 1_000_000.0
            random_rolling = (
                centered_window_ratio(random["cpu"], random["water"], window=3)
                / 1_000_000.0
            )
            inset.plot(
                hours,
                random_hourly,
                color=ORANGE,
                linewidth=0.75,
                marker="o",
                markersize=2.5,
                alpha=0.32,
            )
            inset.plot(hours, random_rolling, color=ORANGE, linewidth=1.8)
            inset.set_xlim(0.0, float(size))
            inset.set_ylim(0.0, 25.0)
            inset.text(
                0.5,
                0.96,
                "Random baseline (0--25 zoom)",
                transform=inset.transAxes,
                ha="center",
                va="top",
                fontsize=7.2,
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
            )
            inset.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
            inset.tick_params(axis="both", labelsize=6.8)
            inset.spines[["top", "right"]].set_visible(False)
        ax.text(
            -0.10,
            1.04,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=INK,
        )

    fig.suptitle("Local computation efficiency across the replay", y=0.98, fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.91,
        "Faint points show independent one-hour bins; bold curves are ratios over centered three-hour windows",
        ha="center",
        fontsize=9.5,
        color="#53606b",
    )
    if comparison_label:
        fig.text(
            0.5,
            0.865,
            comparison_label,
            ha="center",
            fontsize=9,
            color="#53606b",
        )
    fig.text(
        0.5,
        0.02,
        "Higher is preferable. Each ratio uses only jobs completed within that local time window.",
        ha="center",
        fontsize=9,
        color="#53606b",
    )
    top = 0.79 if comparison_label else 0.82
    fig.subplots_adjust(left=0.085, right=0.985, top=top, bottom=0.16, wspace=0.25)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=320)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)

    print(output_stem.with_suffix(".png"))
    print(output_stem.with_suffix(".pdf"))
    print(output_stem.with_suffix(".svg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot non-cumulative normalized-CPU efficiency in hourly windows."
    )
    parser.add_argument("--green", type=Path, default=DEFAULT_GREEN)
    parser.add_argument("--random", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--label",
        default=None,
        help="Optional configuration/date annotation displayed below the subtitle.",
    )
    parser.add_argument(
        "--water-inset",
        action="store_true",
        help="Add a 0--25 zoom inset for the random water-efficiency series.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(
        args.green,
        args.random,
        args.output,
        comparison_label=args.label,
        water_inset=args.water_inset,
    )
