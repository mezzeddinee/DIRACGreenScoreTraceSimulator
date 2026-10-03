from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CI_MAX = 760.0
HI_MAX = 23.84
WC = 0.71
WW = 0.29
GREEN = "#009E73"
ORANGE = "#E69F00"
GRID = "#d8dde1"


def load_hourly(path: Path) -> dict[str, np.ndarray]:
    rows: list[tuple[datetime, datetime, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hi = float(row["assigned_hi_stress_l_per_kwh"])
            if hi <= 0.0:
                raise ValueError(f"Cannot recover facility energy for job {row['job_id']}")
            ci = float(row["assigned_ci_gco2_per_kwh"])

            # Simulator output: water impact = IT energy * PUE * HI.
            facility_energy = float(row["water_impact_stress_l"]) / hi
            envscore_fraction = WC * (1.0 - ci / CI_MAX) + WW * (1.0 - hi / HI_MAX)
            environmental_burden = 1.0 - envscore_fraction
            burden = facility_energy * environmental_burden
            rows.append(
                (
                    datetime.fromisoformat(row["submit_time"]),
                    datetime.fromisoformat(row["finish_time"]),
                    float(row["norm_cpu_seconds"]),
                    burden,
                )
            )

    if not rows:
        raise ValueError(f"No completed jobs in {path}")
    origin = min(row[0] for row in rows)
    bins = np.asarray(
        [max(0, int((row[1] - origin).total_seconds() // 3600)) for row in rows],
        dtype=int,
    )
    size = int(bins.max()) + 1
    return {
        "cpu": np.bincount(bins, weights=[row[2] for row in rows], minlength=size),
        "burden": np.bincount(bins, weights=[row[3] for row in rows], minlength=size),
        "jobs": np.bincount(bins, minlength=size),
    }


def pad(values: np.ndarray, size: int) -> np.ndarray:
    return np.pad(values, (0, size - len(values)), constant_values=0.0)


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def plot(green_path: Path, random_path: Path, output: Path) -> None:
    green = load_hourly(green_path)
    random = load_hourly(random_path)
    size = max(len(green["cpu"]), len(random["cpu"]))
    for dataset in (green, random):
        for key in ("cpu", "burden", "jobs"):
            dataset[key] = pad(dataset[key], size)

    hours = np.arange(size, dtype=float) + 0.5
    kernel = np.ones(3, dtype=float)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.8)
    exported: list[dict[str, object]] = []

    for label, dataset, color in (
        ("GreenScore-based", green, GREEN),
        ("Random", random, ORANGE),
    ):
        hourly = ratio(dataset["cpu"], dataset["burden"]) / 1_000_000.0
        rolling = ratio(
            np.convolve(dataset["cpu"], kernel, mode="same"),
            np.convolve(dataset["burden"], kernel, mode="same"),
        ) / 1_000_000.0
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
        for index in range(size):
            exported.append(
                {
                    "run": label,
                    "hour": index,
                    "hour_midpoint": hours[index],
                    "completed_jobs": int(dataset["jobs"][index]),
                    "norm_cpu_seconds": dataset["cpu"][index],
                    "es_derived_burden_weighted_kwh": dataset["burden"][index],
                    "million_normcpu_per_burden_weighted_kwh": hourly[index],
                    "three_hour_ratio": rolling[index],
                }
            )

    ax.set_title("Computation per ES-derived environmental burden", loc="left", weight="bold")
    ax.set_xlabel("Elapsed simulation time (h)")
    ax.set_ylabel("Million normalized CPU-s / burden-weighted kWh")
    ax.set_xlim(0.0, float(size))
    ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2")
    fig.text(
        0.5,
        0.055,
        "Burden-weighted energy = facility energy × (1 − ES/100). Higher is preferable.",
        ha="center",
        fontsize=8.7,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=320 if suffix == "png" else None)
    plt.close(fig)

    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)

    green_total = green["cpu"].sum() / green["burden"].sum() / 1_000_000.0
    random_total = random["cpu"].sum() / random["burden"].sum() / 1_000_000.0
    print(f"green_total_efficiency={green_total:.6f}")
    print(f"random_total_efficiency={random_total:.6f}")
    print(f"green_change={(green_total / random_total - 1.0) * 100.0:.4f}%")
    for suffix in ("png", "pdf", "svg", "csv"):
        print(output.with_suffix(f".{suffix}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--green", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.green, args.random, args.output)
