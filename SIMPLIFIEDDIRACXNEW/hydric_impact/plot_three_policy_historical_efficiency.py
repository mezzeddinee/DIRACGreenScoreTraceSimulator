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
CARBON_WEIGHT = 0.71
HYDRIC_WEIGHT = 0.29

DELAYED = "#009E73"
NO_DELAY = "#0072B2"
RANDOM = "#E69F00"
GRID = "#d8dde1"
INK = "#26333d"


def load_hourly(path: Path) -> dict[str, np.ndarray | float]:
    rows: list[tuple[datetime, datetime, float, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hi = float(row["assigned_hi_stress_l_per_kwh"])
            if hi <= 0.0:
                raise ValueError(
                    f"Cannot recover facility energy for job {row['job_id']}: HI={hi}"
                )
            ci = float(row["assigned_ci_gco2_per_kwh"])
            water = float(row["water_impact_stress_l"])
            facility_energy = water / hi
            environmental_burden = (
                CARBON_WEIGHT * ci / CI_MAX + HYDRIC_WEIGHT * hi / HI_MAX
            )
            rows.append(
                (
                    datetime.fromisoformat(row["submit_time"]),
                    datetime.fromisoformat(row["finish_time"]),
                    float(row["norm_cpu_seconds"]),
                    float(row["carbon_kg"]),
                    water,
                    facility_energy * environmental_burden,
                )
            )

    if not rows:
        raise ValueError(f"No completed jobs in {path}")
    origin = min(row[0] for row in rows)
    indices = np.asarray(
        [max(0, int((row[1] - origin).total_seconds() // 3600)) for row in rows],
        dtype=int,
    )
    size = int(indices.max()) + 1
    result: dict[str, np.ndarray | float] = {
        "cpu": np.bincount(indices, weights=[row[2] for row in rows], minlength=size),
        "carbon": np.bincount(indices, weights=[row[3] for row in rows], minlength=size),
        "water": np.bincount(indices, weights=[row[4] for row in rows], minlength=size),
        "environmental_burden": np.bincount(
            indices, weights=[row[5] for row in rows], minlength=size
        ),
        "jobs": float(len(rows)),
    }
    for key in ("cpu", "carbon", "water", "environmental_burden"):
        result[f"total_{key}"] = float(np.asarray(result[key]).sum())
    return result


def pad(values: np.ndarray, size: int) -> np.ndarray:
    return np.pad(values, (0, size - values.size), constant_values=0.0)


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def plot(delayed_path: Path, no_delay_path: Path, random_path: Path, output: Path) -> None:
    runs = [
        ("Green, delay 60 min", load_hourly(delayed_path), DELAYED),
        ("Green, no delay", load_hourly(no_delay_path), NO_DELAY),
        ("Random, seed 42", load_hourly(random_path), RANDOM),
    ]
    counts = {int(data["jobs"]) for _, data, _ in runs}
    workloads = [float(data["total_cpu"]) for _, data, _ in runs]
    if len(counts) != 1 or not all(
        np.isclose(workloads[0], value, rtol=1e-10) for value in workloads[1:]
    ):
        raise ValueError("Runs do not contain the same workload")

    size = max(len(np.asarray(data["cpu"])) for _, data, _ in runs)
    for _, data, _ in runs:
        for key in ("cpu", "carbon", "water", "environmental_burden"):
            data[key] = pad(np.asarray(data[key]), size)

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
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    panels = (
        ("carbon", "Carbon efficiency", "Million normalized CPU-s / kgCO$_2$e"),
        ("water", "Water-scarcity efficiency", "Million normalized CPU-s / stress-L"),
        (
            "environmental_burden",
            "Combined environmental-burden efficiency",
            "Million normalized CPU-s / burden-weighted kWh",
        ),
    )
    hours = np.arange(size, dtype=float) + 0.5
    kernel = np.ones(3, dtype=float)

    for panel_index, (impact, title, ylabel) in enumerate(panels):
        ax = axes[panel_index]
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.65, alpha=0.8)
        for label, data, color in runs:
            cpu = np.asarray(data["cpu"])
            burden = np.asarray(data[impact])
            hourly = ratio(cpu, burden) / 1_000_000.0
            rolling = ratio(
                np.convolve(cpu, kernel, mode="same"),
                np.convolve(burden, kernel, mode="same"),
            ) / 1_000_000.0
            ax.plot(hours, hourly, color=color, linewidth=0.75, marker="o",
                    markersize=2.8, alpha=0.23)
            ax.plot(hours, rolling, color=color, linewidth=2.1,
                    label=f"{label} (3-h window)")

        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Elapsed simulation time (h)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, float(size))
        ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2", fontsize=8.2)
        ax.text(-0.11, 1.04, chr(ord("A") + panel_index), transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=INK)

    fig.suptitle("Useful computation per unit environmental metric",
                 y=0.98, fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.915,
        "Same 133,631-job workload; historical-interval CI and WI; local completion-time windows",
        ha="center",
        fontsize=9.5,
        color="#53606b",
    )
    fig.text(
        0.5,
        0.02,
        "Burden = 1 − ES/100 = 0.71(CI/760) + 0.29(WI/23.84). Faint points are one-hour bins; higher is preferable.",
        ha="center",
        fontsize=8.8,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.16, wspace=0.27)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=320 if suffix == ".png" else None)
    plt.close(fig)

    for label, data, _ in runs:
        print(label)
        for impact in ("carbon", "water", "environmental_burden"):
            efficiency = float(data["total_cpu"]) / float(data[f"total_{impact}"]) / 1_000_000.0
            print(
                f"  total_{impact}={float(data[f'total_{impact}']):.12f} "
                f"cpu_per_{impact}_million={efficiency:.6f}"
            )
    for suffix in (".png", ".pdf", ".svg"):
        print(output.with_suffix(suffix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delayed", type=Path, required=True)
    parser.add_argument("--no-delay", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.delayed, args.no_delay, args.random, args.output)
