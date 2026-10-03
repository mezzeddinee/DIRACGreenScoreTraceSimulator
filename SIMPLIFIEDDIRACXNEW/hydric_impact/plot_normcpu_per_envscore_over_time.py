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
            facility_energy = float(row["water_impact_stress_l"]) / hi
            envscore_fraction = WC * (1.0 - ci / CI_MAX) + WW * (1.0 - hi / HI_MAX)
            envscore_weighted_energy = facility_energy * envscore_fraction
            rows.append(
                (
                    datetime.fromisoformat(row["submit_time"]),
                    datetime.fromisoformat(row["finish_time"]),
                    float(row["norm_cpu_seconds"]),
                    envscore_weighted_energy,
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
        "envscore_energy": np.bincount(
            bins, weights=[row[3] for row in rows], minlength=size
        ),
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
        for key in ("cpu", "envscore_energy", "jobs"):
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
        ("Green-hydric", green, GREEN),
        ("Random", random, ORANGE),
    ):
        hourly = ratio(dataset["cpu"], dataset["envscore_energy"]) / 1_000_000.0
        rolling = ratio(
            np.convolve(dataset["cpu"], kernel, mode="same"),
            np.convolve(dataset["envscore_energy"], kernel, mode="same"),
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
                    "envscore_weighted_kwh": dataset["envscore_energy"][index],
                    "million_normcpu_per_envscore_weighted_kwh": hourly[index],
                    "three_hour_ratio": rolling[index],
                }
            )

    ax.set_title("Normalized computation per EnvScore-weighted energy", loc="left", weight="bold")
    ax.set_xlabel("Elapsed simulation time (h)")
    ax.set_ylabel("Million normalized CPU-s / EnvScore-weighted kWh")
    ax.set_xlim(0.0, float(size))
    ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2")
    fig.text(
        0.5,
        0.075,
        "EnvScore/100 = 0.71(1 − CI/760) + 0.29(1 − HI/23.84)",
        ha="center",
        fontsize=8.8,
        color="#53606b",
    )
    fig.text(
        0.5,
        0.040,
        "Faint points: 1-h bins; bold curves: centered 3-h ratios. EnvScore is quality, not burden.",
        ha="center",
        fontsize=8.5,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=320 if suffix == "png" else None)
    plt.close(fig)

    fields = list(exported[0])
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exported)

    for label, dataset in (("green", green), ("random", random)):
        total = dataset["cpu"].sum() / dataset["envscore_energy"].sum() / 1_000_000.0
        print(f"{label}_total_ratio={total:.6f}")
    for suffix in ("png", "pdf", "svg", "csv"):
        print(output.with_suffix(f".{suffix}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--green", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.green, args.random, args.output)
