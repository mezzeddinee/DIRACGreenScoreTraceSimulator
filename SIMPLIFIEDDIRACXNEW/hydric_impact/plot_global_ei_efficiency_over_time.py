from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CI_MAX = 760.0
HI_MAX = 14.70
CARBON_WEIGHT = 0.71
HYDRIC_WEIGHT = 0.29

GREEN = "#009E73"
ORANGE = "#E69F00"
GRID = "#d8dde1"
INK = "#26333d"


def load_jobs(path: Path) -> list[tuple[datetime, datetime, float, float]]:
    jobs: list[tuple[datetime, datetime, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hi = float(row["assigned_hi_stress_l_per_kwh"])
            if hi <= 0.0:
                raise ValueError(
                    f"Cannot recover facility energy for job {row['job_id']}: HI={hi}"
                )
            ci = float(row["assigned_ci_gco2_per_kwh"])
            facility_energy_kwh = float(row["water_impact_stress_l"]) / hi
            ei_intensity = (
                CARBON_WEIGHT * ci / CI_MAX
                + HYDRIC_WEIGHT * hi / HI_MAX
            )
            ei_burden = facility_energy_kwh * ei_intensity
            jobs.append(
                (
                    datetime.fromisoformat(row["submit_time"]),
                    datetime.fromisoformat(row["finish_time"]),
                    float(row["norm_cpu_seconds"]),
                    ei_burden,
                )
            )
    if not jobs:
        raise ValueError(f"No completed jobs found in {path}")
    return jobs


def cumulative(jobs: list[tuple[datetime, datetime, float, float]]) -> dict[str, np.ndarray | float]:
    origin = min(job[0] for job in jobs)
    by_finish: dict[datetime, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for _submit, finish, cpu, burden in jobs:
        by_finish[finish][0] += cpu
        by_finish[finish][1] += burden
    times = sorted(by_finish)
    cpu = np.asarray([by_finish[t][0] for t in times]).cumsum()
    burden = np.asarray([by_finish[t][1] for t in times]).cumsum()
    return {
        "hours": np.asarray([(t - origin).total_seconds() / 3600.0 for t in times]),
        "efficiency": cpu / burden / 1_000_000.0,
        "total_cpu": float(cpu[-1]),
        "total_burden": float(burden[-1]),
    }


def hourly(jobs: list[tuple[datetime, datetime, float, float]], size: int) -> tuple[np.ndarray, np.ndarray]:
    origin = min(job[0] for job in jobs)
    indices = np.asarray(
        [max(0, int((job[1] - origin).total_seconds() // 3600)) for job in jobs],
        dtype=int,
    )
    cpu = np.bincount(indices, weights=[job[2] for job in jobs], minlength=size)
    burden = np.bincount(indices, weights=[job[3] for job in jobs], minlength=size)
    return cpu, burden


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def plot(green_path: Path, random_path: Path, output: Path) -> None:
    green_jobs = load_jobs(green_path)
    random_jobs = load_jobs(random_path)
    green_cumulative = cumulative(green_jobs)
    random_cumulative = cumulative(random_jobs)
    green_origin = min(job[0] for job in green_jobs)
    random_origin = min(job[0] for job in random_jobs)
    size = max(
        int(max((job[1] - green_origin).total_seconds() // 3600 for job in green_jobs)) + 1,
        int(max((job[1] - random_origin).total_seconds() // 3600 for job in random_jobs)) + 1,
    )
    green_cpu, green_burden = hourly(green_jobs, size)
    random_cpu, random_burden = hourly(random_jobs, size)
    kernel = np.ones(3, dtype=float)

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
    runs = (
        ("Green-hydric", green_cumulative, green_cpu, green_burden, GREEN),
        ("Random", random_cumulative, random_cpu, random_burden, ORANGE),
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.65, alpha=0.8)

    for label, data, _cpu, _burden, color in runs:
        x = np.asarray(data["hours"])
        y = np.asarray(data["efficiency"])
        axes[0].plot(x, y, color=color, linewidth=2.0, label=label)
        axes[0].scatter(x[-1], y[-1], color=color, edgecolor="white", s=52, zorder=4)
        axes[0].annotate(
            f"{y[-1]:.1f}",
            (x[-1], y[-1]),
            xytext=(-7, 8 if label == "Green-hydric" else -13),
            textcoords="offset points",
            ha="right",
            va="bottom" if label == "Green-hydric" else "top",
            color=color,
            fontsize=8.5,
            fontweight="semibold",
        )

    hours = np.arange(size, dtype=float) + 0.5
    for label, _data, cpu, burden, color in runs:
        raw = ratio(cpu, burden) / 1_000_000.0
        rolling = ratio(
            np.convolve(cpu, kernel, mode="same"),
            np.convolve(burden, kernel, mode="same"),
        ) / 1_000_000.0
        axes[1].plot(hours, raw, color=color, linewidth=0.8, marker="o", markersize=3.2, alpha=0.28)
        axes[1].plot(hours, rolling, color=color, linewidth=2.2, label=f"{label} (3-h window)")

    axes[0].set_title("Cumulative environmental-burden efficiency", loc="left", weight="bold")
    axes[1].set_title("Local environmental-burden efficiency", loc="left", weight="bold")
    for index, ax in enumerate(axes):
        ax.set_xlabel("Elapsed simulation time (h)")
        ax.set_ylabel("Million normalized CPU-s / burden-weighted kWh")
        ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2")
        ax.text(
            -0.10, 1.04, chr(ord("A") + index), transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=INK,
        )

    green_eff = float(green_cumulative["efficiency"][-1])
    random_eff = float(random_cumulative["efficiency"][-1])
    change = 100.0 * (green_eff / random_eff - 1.0)
    fig.suptitle("Normalized computation per combined environmental burden", y=0.98, fontsize=16, weight="bold")
    fig.text(
        0.5, 0.91,
        f"Burden = 1 − ES/100 = 0.71(CI/760) + 0.29(WI/14.70) · facility-energy weighted · final green change: {change:.2f}%",
        ha="center", fontsize=9.5, color="#53606b",
    )
    fig.text(
        0.5, 0.02,
        "Higher is preferable. Local bold curves use centered three-hour ratios.",
        ha="center", fontsize=9, color="#53606b",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.16, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=320 if suffix == ".png" else None)
    plt.close(fig)

    print(f"green_ei_burden={green_cumulative['total_burden']:.12f}")
    print(f"random_ei_burden={random_cumulative['total_burden']:.12f}")
    print(f"green_efficiency={green_eff:.6f}")
    print(f"random_efficiency={random_eff:.6f}")
    print(f"green_change={change:.4f}%")
    for suffix in (".png", ".pdf", ".svg"):
        print(output.with_suffix(suffix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--green", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.green, args.random, args.output)
