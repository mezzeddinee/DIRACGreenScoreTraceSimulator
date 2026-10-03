from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CF_MAX = 760.0
HI_MAX = 23.84
SOURCE = Path(
    "/home/mezzeddi/PycharmProjects/GreenDIRAC/docs/results/"
    "wattnet_last_week_ci_hi_scaled_2026-08-04_2026-08-10/data/"
    "all_zone_15min_scaled_observations.csv"
)
OUTPUT = Path(__file__).resolve().parent / "advisor_cf_hi_site_histograms_2026-08-04_2026-08-10"
COLORS = ("#0072B2", "#D55E00")


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=320 if suffix == "png" else None)
    plt.close(fig)


def paired_histogram(
    cf: np.ndarray, hi: np.ndarray, title: str, stem: str, ylabel: str
) -> None:
    bins = np.linspace(0.0, 1.0, 21)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for ax, values, label, color in zip(
        axes, (cf, hi), ("Scaled carbon factor", "Scaled hydric impact"), COLORS
    ):
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.7)
        ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean = {values.mean():.4f}")
        ax.axvline(np.median(values), color="#555555", linestyle=":", linewidth=1.2,
                   label=f"Median = {np.median(values):.4f}")
        ax.set_xlabel(label)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="y", color="#d9dde1", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, stem)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, object]] = []
    by_site: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_time: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cf = float(row["ci_gco2e_per_kwh"]) / CF_MAX
            hi = float(row["hi_stress_l_per_kwh"]) / HI_MAX
            item = {
                "timestamp_utc": row["timestamp_utc"],
                "zone": row["zone"],
                "cf_gco2e_per_kwh": row["ci_gco2e_per_kwh"],
                "hi_stress_l_per_kwh": row["hi_stress_l_per_kwh"],
                "cf_scaled": f"{cf:.10f}",
                "hi_scaled": f"{hi:.10f}",
            }
            observations.append(item)
            by_site[row["zone"]].append((cf, hi))
            by_time[row["timestamp_utc"]].append((row["zone"], cf, hi))

    site_rows: list[dict[str, object]] = []
    for site, values in sorted(by_site.items()):
        array = np.asarray(values)
        site_rows.append({
            "zone": site,
            "observations": len(values),
            "mean_cf_scaled": f"{array[:, 0].mean():.10f}",
            "median_cf_scaled": f"{np.median(array[:, 0]):.10f}",
            "mean_hi_scaled": f"{array[:, 1].mean():.10f}",
            "median_hi_scaled": f"{np.median(array[:, 1]):.10f}",
        })
    site_cf = np.asarray([float(row["mean_cf_scaled"]) for row in site_rows])
    site_hi = np.asarray([float(row["mean_hi_scaled"]) for row in site_rows])
    paired_histogram(
        site_cf, site_hi,
        "Distribution of weekly mean scaled impacts across WattNet zones",
        "site_weekly_mean_cf_hi_histograms",
        "Number of WattNet zones",
    )

    # Select a complete snapshot closest to the middle of the analysis period.
    complete = [(timestamp, values) for timestamp, values in by_time.items()
                if len(values) == max(map(len, by_time.values()))]
    middle = datetime.fromisoformat("2026-08-07T12:00:00+00:00")
    snapshot_time, snapshot = min(
        complete,
        key=lambda item: abs((datetime.fromisoformat(item[0].replace("Z", "+00:00")) - middle).total_seconds()),
    )
    snapshot_array = np.asarray([(cf, hi) for _, cf, hi in snapshot])
    paired_histogram(
        snapshot_array[:, 0], snapshot_array[:, 1],
        f"Distribution across WattNet zones at {snapshot_time}",
        "site_snapshot_cf_hi_histograms",
        "Number of WattNet zones",
    )

    all_cf = np.asarray([float(row["cf_scaled"]) for row in observations])
    all_hi = np.asarray([float(row["hi_scaled"]) for row in observations])
    paired_histogram(
        all_cf, all_hi,
        "All WattNet zone–timestamp observations, 4–10 August 2026",
        "pooled_zone_time_cf_hi_histograms",
        "Number of zone–timestamp observations",
    )

    a_rows: list[dict[str, object]] = []
    for timestamp, values in sorted(by_time.items()):
        array = np.asarray([(cf, hi) for _, cf, hi in values])
        mean_cf = array[:, 0].mean()
        mean_hi = array[:, 1].mean()
        a_rows.append({
            "timestamp_utc": timestamp,
            "zone_count": len(values),
            "mean_cf_scaled": f"{mean_cf:.10f}",
            "mean_hi_scaled": f"{mean_hi:.10f}",
            "A_mean_cf_over_mean_hi": f"{mean_cf / mean_hi:.10f}",
        })
    a_values = np.asarray([float(row["A_mean_cf_over_mean_hi"]) for row in a_rows])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(a_values, bins=20, color="#009E73", edgecolor="white")
    ax.axvline(a_values.mean(), color="black", linestyle="--", label=f"Mean = {a_values.mean():.2f}")
    ax.axvline(np.median(a_values), color="#555555", linestyle=":", label=f"Median = {np.median(a_values):.2f}")
    ax.set(title="Distribution of the dynamic correction factor A(t)", xlabel="A(t)", ylabel="Number of 15-min readings")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dde1", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "dynamic_A_histogram")

    write_rows(OUTPUT / "scaled_observations.csv", list(observations[0]), observations)
    write_rows(OUTPUT / "weekly_site_summary.csv", list(site_rows[0]), site_rows)
    write_rows(OUTPUT / "dynamic_A_15min.csv", list(a_rows[0]), a_rows)
    summary = [
        ("analysis_start_utc", min(by_time)),
        ("analysis_end_utc", max(by_time)),
        ("wattnet_zones", len(by_site)),
        ("zone_timestamp_observations", len(observations)),
        ("cf_max_reference_gco2e_per_kwh", CF_MAX),
        ("hi_max_reference_stress_l_per_kwh", HI_MAX),
        ("weekly_site_mean_cf_scaled_mean", site_cf.mean()),
        ("weekly_site_mean_cf_scaled_median", np.median(site_cf)),
        ("weekly_site_mean_hi_scaled_mean", site_hi.mean()),
        ("weekly_site_mean_hi_scaled_median", np.median(site_hi)),
        ("A_from_all_observations", all_cf.mean() / all_hi.mean()),
        ("dynamic_A_mean", a_values.mean()),
        ("dynamic_A_median", np.median(a_values)),
        ("dynamic_A_min", a_values.min()),
        ("dynamic_A_max", a_values.max()),
        ("representative_snapshot_utc", snapshot_time),
    ]
    write_rows(OUTPUT / "summary_statistics.csv", ["metric", "value"],
               [{"metric": key, "value": value} for key, value in summary])
    (OUTPUT / "README.txt").write_text(
        "CF and HI histograms requested by Andrei\n\n"
        "Normalization:\n"
        "  CF_scaled = CF / 760 gCO2e/kWh\n"
        "  HI_scaled = HI / 23.84 stress-L/kWh\n\n"
        "The reference maxima are common across WattNet zones. The main figure is "
        "site_weekly_mean_cf_hi_histograms.pdf. Snapshot and pooled versions are "
        "included as sensitivity views. dynamic_A_histogram.pdf shows "
        "A(t) = mean(CF_scaled(t)) / mean(HI_scaled(t)).\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
