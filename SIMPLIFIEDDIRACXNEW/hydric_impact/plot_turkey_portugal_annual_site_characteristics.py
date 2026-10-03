from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent.parent
SOURCE = (
    WORKSPACE
    / "alvaro_monthly_ei_es_gs_2025-06_2026-07"
    / "monthly_environmental_score_components.csv"
)
FIGURE_DIR = WORKSPACE / "DIRAC_simulation_reproduction_guide_figures"
DATA_DIR = WORKSPACE / "DIRAC_simulation_reproduction_guide_data"

SITE_ORDER = (
    "IN2P3-IRES",
    "SARA-MATRIX",
    "NCG-INGRID-PT",
    "TR-03-METU",
)
SITE_CONTEXT = {
    "IN2P3-IRES": ("France", "FR", "#0072B2"),
    "SARA-MATRIX": ("Netherlands", "NL", "#009E73"),
    "NCG-INGRID-PT": ("Portugal", "PT", "#E69F00"),
    "TR-03-METU": ("Turkey", "TR", "#D55E00"),
}
MONTHS = tuple(
    [f"2025-{month:02d}" for month in range(6, 13)]
    + [f"2026-{month:02d}" for month in range(1, 6)]
)


def load_annual_means() -> list[dict[str, object]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    results: list[dict[str, object]] = []
    for site in SITE_ORDER:
        rows = [
            row
            for row in source_rows
            if row["site"] == site and row["month"] in MONTHS
        ]
        rows.sort(key=lambda row: row["month"])
        if [row["month"] for row in rows] != list(MONTHS):
            raise ValueError(f"Incomplete 12-month data for {site}")

        country, expected_zone, _ = SITE_CONTEXT[site]
        zones = {row["wattnet_zone"] for row in rows}
        if zones != {expected_zone}:
            raise ValueError(f"Unexpected WattNet zone for {site}: {zones}")

        paired_samples = sum(int(row["paired_15min_samples"]) for row in rows)
        if paired_samples <= 0:
            raise ValueError(f"No paired WattNet observations for {site}")

        mean_ci = sum(
            float(row["mean_cf_gco2e_per_kwh"])
            * int(row["paired_15min_samples"])
            for row in rows
        ) / paired_samples
        mean_wi = sum(
            float(row["mean_hi_stress_l_per_kwh"])
            * int(row["paired_15min_samples"])
            for row in rows
        ) / paired_samples
        mean_cee = sum(float(row["monthly_cee"]) for row in rows) / len(rows)
        observed_months = sum(
            row["cee_source"] == "observed_elasticsearch" for row in rows
        )

        results.append(
            {
                "site": site,
                "country": country,
                "wattnet_zone": expected_zone,
                "period_start": "2025-06",
                "period_end": "2026-05",
                "paired_valid_complete_15min_samples": paired_samples,
                "mean_ci_gco2e_per_kwh": mean_ci,
                "mean_wi_stress_l_per_kwh": mean_wi,
                "mean_cee": mean_cee,
                "observed_cee_months": observed_months,
                "imputed_cee_months": len(rows) - observed_months,
            }
        )
    return results


def write_summary(rows: list[dict[str, object]]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "turkey_portugal_annual_site_characteristics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["mean_ci_gco2e_per_kwh"] = f"{float(row['mean_ci_gco2e_per_kwh']):.8f}"
            formatted["mean_wi_stress_l_per_kwh"] = f"{float(row['mean_wi_stress_l_per_kwh']):.8f}"
            formatted["mean_cee"] = f"{float(row['mean_cee']):.8f}"
            writer.writerow(formatted)
    return output


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman", "serif"],
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_value_labels(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    maximum: float,
    decimals: int,
) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            value + maximum * 0.018,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.{decimals}f}",
            ha="left",
            va="center",
            fontsize=9.0,
            color="#20252a",
        )


def plot(rows: list[dict[str, object]]) -> Path:
    configure_style()
    y = np.arange(len(rows), dtype=float)
    labels = [f"{row['site']}\n({row['country']})" for row in rows]
    colors = [SITE_CONTEXT[str(row["site"])][2] for row in rows]

    panels = (
        (
            np.asarray([float(row["mean_ci_gco2e_per_kwh"]) for row in rows]),
            "A  Carbon intensity",
            "gCO$_2$e kWh$^{-1}$",
            "Lower is preferable",
            1,
        ),
        (
            np.asarray([float(row["mean_wi_stress_l_per_kwh"]) for row in rows]),
            "B  Water-scarcity intensity",
            "stress-L kWh$^{-1}$",
            "Lower is preferable",
            3,
        ),
        (
            np.asarray([float(row["mean_cee"]) for row in rows]),
            "C  CPU energy efficiency",
            "Normalized CPU performance\nper watt",
            "Higher is preferable",
            2,
        ),
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.6), sharey=True)
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        values, title, xlabel, direction, decimals = panel
        maximum = float(np.max(values))
        bars = ax.barh(
            y,
            values,
            height=0.58,
            color=colors,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.set_xlim(0.0, maximum * 1.25)
        ax.set_title(title, loc="left", weight="bold", pad=19)
        ax.text(
            0.0,
            1.015,
            direction,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#59636d",
        )
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#d8dde2", linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        add_value_labels(ax, bars, values, maximum, int(decimals))
        if index == 0:
            ax.set_yticks(y, labels)
        else:
            ax.tick_params(labelleft=False)

    axes[0].invert_yaxis()
    fig.suptitle(
        "Annual-average characteristics of the sensitivity sites",
        y=0.985,
        fontsize=14.0,
        weight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "June 2025--May 2026; WattNet operational global-coverage signals and DIRAC accounting CEE",
        ha="center",
        fontsize=9.0,
        color="#59636d",
    )
    fig.subplots_adjust(left=0.205, right=0.985, top=0.78, bottom=0.19, wspace=0.38)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "turkey_portugal_annual_site_characteristics"
    fig.savefig(stem.with_suffix(".png"), dpi=320)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    return stem.with_suffix(".pdf")


def main() -> None:
    rows = load_annual_means()
    csv_path = write_summary(rows)
    figure_path = plot(rows)
    print(csv_path)
    print(figure_path)
    for row in rows:
        print(
            f"{row['site']}: CI={float(row['mean_ci_gco2e_per_kwh']):.3f}, "
            f"WI={float(row['mean_wi_stress_l_per_kwh']):.4f}, "
            f"CEE={float(row['mean_cee']):.3f}"
        )


if __name__ == "__main__":
    main()
