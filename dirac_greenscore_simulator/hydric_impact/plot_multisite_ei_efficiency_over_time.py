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
DEFAULT_GREEN = BASE / "timeseries/greenhydric_multisite_new_gs/completed_jobs.csv"
DEFAULT_RANDOM = BASE / "timeseries/random_multisite_new_gs/completed_jobs.csv"
DEFAULT_OUTPUT = BASE / "timeseries/new_gs_comparison/normcpu_per_ei_over_time"

# Fixed valid/complete WattNet maxima over June 2025--July 2026.
CI_MAX_GCO2_PER_KWH = 760.0
HI_MAX_STRESS_L_PER_KWH = 23.84

# Ratio of the 14 equally weighted monthly means of Cscaled and Hscaled.
K_REFERENCE = 14.870854589863225
CARBON_WEIGHT = 0.71
HYDRIC_WEIGHT = 0.29

GREEN = "#009E73"
ORANGE = "#E69F00"
INK = "#26333d"
GRID = "#d8dde1"


def load_hourly(path: Path, k_factor: float) -> dict[str, np.ndarray]:
    rows: list[tuple[datetime, datetime, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            energy = float(row["total_energy_kwh"])
            hi = float(row["assigned_hi_stress_l_per_kwh"])
            water_impact = float(row["water_impact_stress_l"])
            ci = float(row["assigned_ci_gco2_per_kwh"])

            # water_impact = IT energy * PUE * HI. Recover the facility energy
            # actually used by the run, avoiding assumptions about static PUE.
            if hi > 0.0:
                facility_energy = water_impact / hi
            else:
                raise ValueError(f"Cannot recover facility energy for job {row['job_id']}")

            c_scaled = ci / CI_MAX_GCO2_PER_KWH
            h_scaled = hi / HI_MAX_STRESS_L_PER_KWH
            ei_intensity = (
                CARBON_WEIGHT * c_scaled
                + HYDRIC_WEIGHT * k_factor * h_scaled
            )
            ei_burden = facility_energy * ei_intensity
            rows.append(
                (
                    datetime.fromisoformat(row["submit_time"]),
                    datetime.fromisoformat(row["finish_time"]),
                    float(row["norm_cpu_seconds"]),
                    ei_burden,
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
    return {
        "cpu": np.bincount(hour_index, weights=[row[2] for row in rows], minlength=size),
        "ei": np.bincount(hour_index, weights=[row[3] for row in rows], minlength=size),
        "jobs": np.bincount(hour_index, minlength=size),
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


def centered_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    kernel = np.ones(3, dtype=float)
    return ratio(
        np.convolve(numerator, kernel, mode="same"),
        np.convolve(denominator, kernel, mode="same"),
    )


def plot(green_path: Path, random_path: Path, output: Path) -> None:
    configurations = [
        ("Without distribution balancing ($K=1$)", 1.0),
        (f"With fixed distribution balancing ($K={K_REFERENCE:.2f}$)", K_REFERENCE),
    ]
    data = []
    for title, k_factor in configurations:
        green = load_hourly(green_path, k_factor)
        random = load_hourly(random_path, k_factor)
        size = max(len(green["cpu"]), len(random["cpu"]))
        for dataset in (green, random):
            for key in ("cpu", "ei", "jobs"):
                dataset[key] = pad(dataset[key], size)
        data.append((title, k_factor, green, random, size))

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
    for index, (title, _k, green, random, size) in enumerate(data):
        ax = axes[index]
        hours = np.arange(size, dtype=float) + 0.5
        ax.set_facecolor("white")
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
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Elapsed simulation time (h)")
        ax.set_ylabel("Million normalized CPU-s / EI-weighted kWh")
        ax.set_xlim(0.0, float(size))
        ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2")
        ax.text(-0.10, 1.04, chr(ord("A") + index), transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=INK)

    fig.suptitle("Local computation delivered per combined environmental burden",
                 y=0.98, fontsize=15.5, weight="bold")
    fig.text(
        0.5, 0.91,
        "Fixed 14-month references: CImax = 760 gCO$_2$e/kWh; "
        "HImax = 23.84 stress-L/kWh",
        ha="center", fontsize=9.5, color="#53606b",
    )
    fig.text(
        0.5, 0.02,
        "Faint points are one-hour bins; bold curves are centered three-hour ratios. Higher is preferable.",
        ha="center", fontsize=9, color="#53606b",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.16, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=320)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)

    for title, k_factor, green, random, _size in data:
        green_eff = green["cpu"].sum() / green["ei"].sum() / 1_000_000.0
        random_eff = random["cpu"].sum() / random["ei"].sum() / 1_000_000.0
        print(
            f"{title}: green={green_eff:.6f}, random={random_eff:.6f}, "
            f"change={(green_eff / random_eff - 1.0) * 100.0:.4f}%"
        )
    for suffix in ("png", "pdf", "svg"):
        print(output.with_suffix(f".{suffix}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--green", type=Path, default=DEFAULT_GREEN)
    parser.add_argument("--random", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.green, args.random, args.output)
