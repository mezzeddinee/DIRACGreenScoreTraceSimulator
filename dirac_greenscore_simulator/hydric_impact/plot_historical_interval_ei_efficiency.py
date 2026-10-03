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

GREEN = "#009E73"
ORANGE = "#E69F00"
GRID = "#d8dde1"
INK = "#26333d"


def load_hourly(path: Path) -> dict[str, np.ndarray | float]:
    rows: list[tuple[datetime, datetime, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hi = float(row["assigned_hi_stress_l_per_kwh"])
            if hi <= 0.0:
                raise ValueError(
                    f"Cannot recover facility energy for job {row['job_id']}: HI={hi}"
                )
            ci = float(row["assigned_ci_gco2_per_kwh"])
            facility_energy = float(row["water_impact_stress_l"]) / hi
            environmental_burden = (
                CARBON_WEIGHT * ci / CI_MAX + HYDRIC_WEIGHT * hi / HI_MAX
            )
            rows.append(
                (
                    datetime.fromisoformat(row["submit_time"]),
                    datetime.fromisoformat(row["finish_time"]),
                    float(row["norm_cpu_seconds"]),
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
    cpu = np.bincount(indices, weights=[row[2] for row in rows], minlength=size)
    burden = np.bincount(indices, weights=[row[3] for row in rows], minlength=size)
    return {
        "cpu": cpu,
        "burden": burden,
        "total_cpu": float(cpu.sum()),
        "total_burden": float(burden.sum()),
        "jobs": float(len(rows)),
    }


def pad(values: np.ndarray, size: int) -> np.ndarray:
    return np.pad(values, (0, size - values.size), constant_values=0.0)


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
    if int(green["jobs"]) != int(random["jobs"]):
        raise ValueError("Runs contain different job counts")
    if not np.isclose(green["total_cpu"], random["total_cpu"], rtol=1e-10):
        raise ValueError("Runs contain different normalized CPU workloads")

    size = max(len(green["cpu"]), len(random["cpu"]))
    for data in (green, random):
        data["cpu"] = pad(np.asarray(data["cpu"]), size)
        data["burden"] = pad(np.asarray(data["burden"]), size)

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
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.8)
    hours = np.arange(size, dtype=float) + 0.5
    kernel = np.ones(3, dtype=float)

    for label, data, color in (
        ("Green-hydric, 60-min delay", green, GREEN),
        ("Random, seed 42", random, ORANGE),
    ):
        cpu = np.asarray(data["cpu"])
        burden = np.asarray(data["burden"])
        hourly = ratio(cpu, burden) / 1_000_000.0
        rolling = ratio(
            np.convolve(cpu, kernel, mode="same"),
            np.convolve(burden, kernel, mode="same"),
        ) / 1_000_000.0
        ax.plot(hours, hourly, color=color, linewidth=0.8, marker="o",
                markersize=3.2, alpha=0.28)
        ax.plot(hours, rolling, color=color, linewidth=2.2,
                label=f"{label} (3-h window)")

    ax.set_title("Normalized computation per combined environmental burden",
                 loc="left", weight="bold")
    ax.set_xlabel("Elapsed simulation time (h)")
    ax.set_ylabel("Million normalized CPU-s / burden-weighted kWh")
    ax.set_xlim(0.0, float(size))
    ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2")
    fig.text(
        0.5,
        0.91,
        "Burden = 1 − ES/100 = 0.71(CI/760) + 0.29(WI/23.84)",
        ha="center",
        fontsize=9.2,
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
    fig.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=320 if suffix == ".png" else None)
    plt.close(fig)

    green_eff = float(green["total_cpu"]) / float(green["total_burden"]) / 1_000_000.0
    random_eff = float(random["total_cpu"]) / float(random["total_burden"]) / 1_000_000.0
    print(f"green_environmental_burden={green['total_burden']:.12f}")
    print(f"random_environmental_burden={random['total_burden']:.12f}")
    print(f"green_environmental_burden_efficiency={green_eff:.6f}")
    print(f"random_environmental_burden_efficiency={random_eff:.6f}")
    print(f"green_efficiency_change={(green_eff / random_eff - 1.0) * 100.0:.4f}%")
    for suffix in (".png", ".pdf", ".svg"):
        print(output.with_suffix(suffix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--green", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.green, args.random, args.output)
