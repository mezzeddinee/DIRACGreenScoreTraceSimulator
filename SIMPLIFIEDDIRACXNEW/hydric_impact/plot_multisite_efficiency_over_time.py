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


BASE = Path(__file__).resolve().parent
DEFAULT_GREEN = BASE / "timeseries" / "green_multisite" / "completed_jobs.csv"
DEFAULT_RANDOM = BASE / "timeseries" / "random_multisite" / "completed_jobs.csv"
DEFAULT_OUTPUT = BASE / "multisite_normcpu_efficiency_over_time"

GREEN = "#009E73"
ORANGE = "#E69F00"
INK = "#26333d"
GRID = "#d8dde1"


def load_cumulative(path: Path) -> dict[str, np.ndarray | float | datetime]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing job-level run data: {path}\n"
            "Run the simulator with SIMULATOR_RUN_LABEL=green_multisite and "
            "SIMULATOR_RUN_LABEL=random_multisite first."
        )

    by_finish: dict[datetime, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    first_submit: datetime | None = None
    job_count = 0

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            submit = datetime.fromisoformat(row["submit_time"])
            finish = datetime.fromisoformat(row["finish_time"])
            first_submit = submit if first_submit is None else min(first_submit, submit)
            bucket = by_finish[finish]
            bucket[0] += float(row["norm_cpu_seconds"])
            bucket[1] += float(row["carbon_kg"])
            bucket[2] += float(row["water_impact_stress_l"])
            job_count += 1

    if first_submit is None or not by_finish:
        raise ValueError(f"No completed jobs found in {path}")

    finish_times = sorted(by_finish)
    norm_cpu = np.asarray([by_finish[t][0] for t in finish_times]).cumsum()
    carbon = np.asarray([by_finish[t][1] for t in finish_times]).cumsum()
    water = np.asarray([by_finish[t][2] for t in finish_times]).cumsum()
    elapsed_hours = np.asarray(
        [(finish - first_submit).total_seconds() / 3600.0 for finish in finish_times]
    )

    carbon_efficiency = np.divide(
        norm_cpu,
        carbon,
        out=np.full_like(norm_cpu, np.nan),
        where=carbon > 0.0,
    )
    water_efficiency = np.divide(
        norm_cpu,
        water,
        out=np.full_like(norm_cpu, np.nan),
        where=water > 0.0,
    )

    return {
        "elapsed_hours": elapsed_hours,
        "carbon_efficiency": carbon_efficiency,
        "water_efficiency": water_efficiency,
        "total_norm_cpu": float(norm_cpu[-1]),
        "total_carbon": float(carbon[-1]),
        "total_water": float(water[-1]),
        "job_count": float(job_count),
    }


def final_value(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float("nan")


def plot(green_path: Path, random_path: Path, output_stem: Path) -> None:
    green = load_cumulative(green_path)
    random = load_cumulative(random_path)

    green_jobs = int(green["job_count"])
    random_jobs = int(random["job_count"])
    if green_jobs != random_jobs:
        raise ValueError(f"Run job counts differ: green={green_jobs}, random={random_jobs}")
    if not np.isclose(float(green["total_norm_cpu"]), float(random["total_norm_cpu"]), rtol=1e-10):
        raise ValueError("Runs do not contain the same total normalized CPU workload")

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    panels = [
        (
            "carbon_efficiency",
            "Carbon efficiency over time",
            "Million normalized CPU-s / kgCO$_2$e",
        ),
        (
            "water_efficiency",
            "Water-scarcity efficiency over time",
            "Million normalized CPU-s / stress-L",
        ),
    ]

    for index, (key, title, ylabel) in enumerate(panels):
        ax = axes[index]
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.65, alpha=0.8)

        for label, data, color in [
            ("Green-hydric", green, GREEN),
            ("Random", random, ORANGE),
        ]:
            x = np.asarray(data["elapsed_hours"])
            y = np.asarray(data[key]) / 1_000_000.0
            ax.plot(x, y, color=color, linewidth=2.0, label=label)
            endpoint = final_value(y)
            ax.scatter(x[-1], endpoint, color=color, edgecolor="white", linewidth=0.7, s=52, zorder=4)
            ax.annotate(
                f"{endpoint:,.1f}",
                (x[-1], endpoint),
                xytext=(-7, 7 if label == "Green-hydric" else -12),
                textcoords="offset points",
                ha="right",
                va="bottom" if label == "Green-hydric" else "top",
                fontsize=8.5,
                color=color,
                fontweight="semibold",
            )

        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Elapsed simulation time (h)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2")
        ax.text(
            -0.10,
            1.04,
            chr(ord("A") + index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=INK,
        )

    green_carbon_eff = final_value(np.asarray(green["carbon_efficiency"]))
    random_carbon_eff = final_value(np.asarray(random["carbon_efficiency"]))
    green_water_eff = final_value(np.asarray(green["water_efficiency"]))
    random_water_eff = final_value(np.asarray(random["water_efficiency"]))
    carbon_gain = 100.0 * (green_carbon_eff / random_carbon_eff - 1.0)
    water_gain = 100.0 * (green_water_eff / random_water_eff - 1.0)

    fig.suptitle("Useful computation per unit environmental impact", y=0.98, fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.91,
        (
            f"Same {green_jobs:,}-job workload · cumulative completed normalized CPU · "
            f"final efficiency gain: {carbon_gain:.2f}% carbon, {water_gain:.2f}% water"
        ),
        ha="center",
        fontsize=9.5,
        color="#53606b",
    )
    fig.text(
        0.5,
        0.02,
        "Higher is preferable. Ratios are cumulative through each job-completion time.",
        ha="center",
        fontsize=9,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.16, wspace=0.25)

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
        description="Plot cumulative normalized-CPU efficiency for green and random multi-site runs."
    )
    parser.add_argument("--green", type=Path, default=DEFAULT_GREEN)
    parser.add_argument("--random", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.green, args.random, args.output)
