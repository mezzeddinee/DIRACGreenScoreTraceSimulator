from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "advisor_cf_hi_site_histograms_2026-08-04_2026-08-10"
INPUT = ROOT / "scaled_observations.csv"
OUTPUT = ROOT / "pooled_cf_hi_corrected_distribution"
BIN_WIDTH = 0.05


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cf = np.asarray([float(row["cf_scaled"]) for row in rows])
    hi = np.asarray([float(row["hi_scaled"]) for row in rows])
    factor = cf.mean() / hi.mean()
    corrected = factor * hi

    panels = (
        (cf, "Scaled carbon factor", "#0072B2", 1.0),
        (hi, "Scaled hydric impact", "#D55E00", 1.0),
        (corrected, f"Corrected hydric impact: A × HIscaled\n(A = {factor:.4f})", "#009E73", 1.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, (values, label, color, fixed_max) in zip(axes, panels):
        upper = fixed_max if fixed_max is not None else np.ceil(values.max() / BIN_WIDTH) * BIN_WIDTH
        bins = np.arange(0.0, upper + BIN_WIDTH * 1.01, BIN_WIDTH)
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean = {values.mean():.4f}")
        ax.axvline(np.median(values), color="#555555", linestyle=":", linewidth=1.2,
                   label=f"Median = {np.median(values):.4f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of zone–timestamp observations")
        ax.set_xlim(0.0, upper)
        ax.grid(axis="y", color="#d9dde1", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.3)
    axes[2].text(
        0.98, 0.72,
        f">1 and not displayed: {np.count_nonzero(corrected > 1):,} observations "
        f"({np.mean(corrected > 1) * 100:.1f}%)",
        transform=axes[2].transAxes, ha="right", fontsize=8.5, color="#44505a",
    )
    fig.suptitle(
        "Pooled WattNet distributions before and after hydric-impact correction",
        fontweight="bold",
    )
    fig.tight_layout()
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=320 if suffix == "png" else None)
    plt.close(fig)

    with (ROOT / "corrected_hi_observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp_utc", "zone", "cf_scaled", "hi_scaled", "A", "A_times_hi_scaled"],
        )
        writer.writeheader()
        for row, c, h, adjusted in zip(rows, cf, hi, corrected):
            writer.writerow({
                "timestamp_utc": row["timestamp_utc"], "zone": row["zone"],
                "cf_scaled": f"{c:.10f}", "hi_scaled": f"{h:.10f}",
                "A": f"{factor:.10f}", "A_times_hi_scaled": f"{adjusted:.10f}",
            })
    with (ROOT / "corrected_hi_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in (
            ("observations", len(cf)), ("mean_cf_scaled", cf.mean()),
            ("mean_hi_scaled", hi.mean()), ("A", factor),
            ("mean_A_times_hi_scaled", corrected.mean()),
            ("median_A_times_hi_scaled", np.median(corrected)),
            ("max_A_times_hi_scaled", corrected.max()),
            ("observations_A_times_hi_above_1", np.count_nonzero(corrected > 1)),
            ("fraction_A_times_hi_above_1", np.mean(corrected > 1)),
            ("histogram_bin_width", BIN_WIDTH),
        ):
            writer.writerow([metric, value])
    print(f"A={factor:.10f}")
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
