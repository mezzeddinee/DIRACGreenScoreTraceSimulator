from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from fetch_plot_wattnet_week import DIRAC_SITE_ZONES


BASE = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    BASE / "wattnet_last_complete_week_all_zones" / "wattnet_ci_hi_15min.csv"
)
DEFAULT_SITES = BASE.parent / "sitesnew.csv"
DEFAULT_OUTPUT = (
    BASE / "wattnet_last_complete_week_all_zones" / "dirac_ei_daily_minmax"
)
CI = "ci_gco2_per_kwh"
HI = "hi_stress_l_per_kwh"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_final(row: dict[str, str], value_name: str) -> bool:
    return (
        row[f"{value_name}_valid"] == "True"
        and row[f"{value_name}_zone_status"] == "complete"
    )


def load_site_mapping(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    missing = [row["site"] for row in rows if row["site"] not in DIRAC_SITE_ZONES]
    if missing:
        raise ValueError(f"No WattNet zone mapping for DIRAC sites: {missing}")
    return [
        {
            "site": row["site"],
            "wattnet_zone": DIRAC_SITE_ZONES[row["site"]],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
        }
        for row in rows
    ]


def scale(value: float, minimum: float, maximum: float) -> float:
    span = maximum - minimum
    if span <= 0.0:
        raise ValueError("Cannot apply min-max scaling to a zero-width range")
    return (value - minimum) / span


def plot_day(
    day: str,
    zone_values: dict[str, dict[str, list[float]]],
    zone_sites: dict[str, list[str]],
    reference: dict[str, float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7.8))
    zones = sorted(zone_values)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(zones)))
    for zone, color in zip(zones, colors):
        sites = zone_sites[zone]
        label = f"{zone}: {sites[0]}" if len(sites) == 1 else f"{zone} ({len(sites)} DIRAC sites)"
        ax.plot(
            zone_values[zone]["hour"],
            zone_values[zone]["ei"],
            color=color,
            linewidth=2.1,
            label=label,
        )

    ax.set_title(
        "EI = 0.71 × Cscaled + 0.29 × Hscaled",
        loc="left",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Environmental-impact index, EI (higher is worse)")
    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.grid(True, color="#d8dde1", linewidth=0.7, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.suptitle(
        f"DIRAC-site EI with daily min–max scaling — {day} UTC",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.052,
        "Cscaled = (CI − CImin) / (CImax − CImin)     Hscaled = (HI − HImin) / (HImax − HImin)",
        ha="center",
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.025,
        (
            f"All-zone daily references: CI [{reference['ci_min']:.2f}, {reference['ci_max']:.2f}] gCO2/kWh; "
            f"HI [{reference['hi_min']:.2f}, {reference['hi_max']:.2f}] stress-L/kWh. Valid, complete observations only."
        ),
        ha="center",
        fontsize=8.8,
    )
    fig.subplots_adjust(left=0.08, right=0.75, top=0.88, bottom=0.15)
    return fig


def run(source: Path, sites_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(source)
    site_mapping = load_site_mapping(sites_path)
    zone_sites: dict[str, list[str]] = defaultdict(list)
    for mapping in site_mapping:
        zone_sites[mapping["wattnet_zone"]].append(mapping["site"])

    write_csv(
        output / "dirac_site_zone_mapping.csv",
        site_mapping,
        ["site", "wattnet_zone", "latitude", "longitude"],
    )

    output_rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    png_paths: list[Path] = []
    pdf_path = output / "dirac_ei_minmax_daily_curves.pdf"
    days = sorted({row["date_utc"] for row in rows})

    with PdfPages(pdf_path) as pdf:
        for day in days:
            day_rows = [
                row for row in rows
                if row["date_utc"] == day and is_final(row, CI) and is_final(row, HI)
            ]
            ci_values = np.asarray([float(row[CI]) for row in day_rows])
            hi_values = np.asarray([float(row[HI]) for row in day_rows])
            reference = {
                "ci_min": float(ci_values.min()),
                "ci_max": float(ci_values.max()),
                "hi_min": float(hi_values.min()),
                "hi_max": float(hi_values.max()),
            }
            references.append({"date_utc": day, **reference})

            zone_values: dict[str, dict[str, list[float]]] = {}
            selected = [row for row in day_rows if row["zone"] in zone_sites]
            selected.sort(key=lambda row: (row["zone"], row["timestamp_utc"]))
            for row in selected:
                zone = row["zone"]
                timestamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
                cscaled = scale(float(row[CI]), reference["ci_min"], reference["ci_max"])
                hscaled = scale(float(row[HI]), reference["hi_min"], reference["hi_max"])
                ei = 0.71 * cscaled + 0.29 * hscaled
                values = zone_values.setdefault(zone, {"hour": [], "ei": []})
                values["hour"].append(timestamp.hour + timestamp.minute / 60.0)
                values["ei"].append(ei)
                for site in zone_sites[zone]:
                    output_rows.append(
                        {
                            "timestamp_utc": row["timestamp_utc"],
                            "date_utc": day,
                            "time_utc": row["time_utc"],
                            "site": site,
                            "wattnet_zone": zone,
                            "ci_gco2_per_kwh": row[CI],
                            "hi_stress_l_per_kwh": row[HI],
                            "ci_min_gco2_per_kwh": reference["ci_min"],
                            "ci_max_gco2_per_kwh": reference["ci_max"],
                            "hi_min_stress_l_per_kwh": reference["hi_min"],
                            "hi_max_stress_l_per_kwh": reference["hi_max"],
                            "cscaled": cscaled,
                            "hscaled": hscaled,
                            "ei": ei,
                        }
                    )

            figure = plot_day(day, zone_values, zone_sites, reference)
            png_path = output / f"dirac_ei_minmax_{day}.png"
            figure.savefig(png_path, dpi=220)
            pdf.savefig(figure)
            plt.close(figure)
            png_paths.append(png_path)

    write_csv(
        output / "dirac_ei_minmax_15min.csv",
        output_rows,
        [
            "timestamp_utc", "date_utc", "time_utc", "site", "wattnet_zone",
            "ci_gco2_per_kwh", "hi_stress_l_per_kwh",
            "ci_min_gco2_per_kwh", "ci_max_gco2_per_kwh",
            "hi_min_stress_l_per_kwh", "hi_max_stress_l_per_kwh",
            "cscaled", "hscaled", "ei",
        ],
    )
    write_csv(
        output / "daily_minmax_references.csv",
        references,
        ["date_utc", "ci_min", "ci_max", "hi_min", "hi_max"],
    )
    print(pdf_path)
    print(output / "dirac_ei_minmax_15min.csv")
    print(output / "daily_minmax_references.csv")
    for path in png_paths:
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot DIRAC-site EI using daily all-zone min-max scaling."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.source, args.sites, args.output)
