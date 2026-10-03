from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))

from ci_provider import MidpointCIProvider


WEEK = BASE / "wattnet_last_complete_week_all_zones"
DEFAULT_MAX_NORM = WEEK / "wattnet_dirac_ei_15min.csv"
DEFAULT_MINMAX = WEEK / "dirac_ei_daily_minmax" / "dirac_ei_minmax_15min.csv"
DEFAULT_SITES = ROOT / "sitesnew.csv"
DEFAULT_CONFIG = ROOT / "cim.conf"
DEFAULT_OUTPUT = WEEK / "dirac_green_score_comparison"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_site_parameters(sites_path: Path, config_path: Path) -> dict[str, dict[str, Any]]:
    provider = MidpointCIProvider.from_config(config_path)
    parameters: dict[str, dict[str, Any]] = {}
    for row in read_csv(sites_path):
        site = row["site"]
        tdp = float(row["avg_tdp_w"])
        cores = float(row["avg_total_cores"])
        cpu_normalization = float(row["perf_hs06"])
        if tdp <= 0.0:
            raise ValueError(f"Non-positive TDP for site={site}")
        pue = provider.get_pue(
            site, float(row["latitude"]), float(row["longitude"])
        )
        cee = cpu_normalization * cores / tdp
        parameters[site] = {
            "site": site,
            "avg_tdp_w": tdp,
            "avg_total_cores": cores,
            "cpu_normalization_factor": cpu_normalization,
            "pue": pue,
            "cee": cee,
        }
    return parameters


def keyed(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["timestamp_utc"], row["site"]): row for row in rows}


def green_score(cee: float, ei: float, pue: float) -> float:
    denominator = ei * pue
    return cee / denominator if denominator > 0.0 else math.nan


def plot_day(
    day: str,
    rows: list[dict[str, Any]],
    sites: list[str],
    y_max: float,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(sites)))
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_site[row["site"]].append(row)

    for site, color in zip(sites, colors):
        site_rows = sorted(by_site[site], key=lambda row: row["timestamp_utc"])
        hours = [
            int(row["time_utc"][:2]) + int(row["time_utc"][3:]) / 60.0
            for row in site_rows
        ]
        axes[0].plot(
            hours,
            [row["green_score_max_norm"] for row in site_rows],
            color=color,
            linewidth=1.55,
            label=site,
        )
        axes[1].plot(
            hours,
            [row["green_score_minmax"] for row in site_rows],
            color=color,
            linewidth=1.55,
            label=site,
        )

    axes[0].set_title(
        "Maximum normalization: Cscaled = CI / CImax; Hscaled = HI / HImax",
        loc="left",
        fontsize=12.5,
        weight="bold",
    )
    axes[1].set_title(
        "Min–max normalization: Cscaled = (CI − CImin)/(CImax − CImin); Hscaled = (HI − HImin)/(HImax − HImin)",
        loc="left",
        fontsize=12.5,
        weight="bold",
    )
    axes[1].set_xlabel("UTC time")
    axes[1].set_xlim(0.0, 24.0)
    axes[1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    for ax in axes:
        ax.set_ylabel("Green Score (higher is better)")
        ax.set_ylim(0.0, y_max)
        ax.grid(True, color="#d8dde1", linewidth=0.65, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        fontsize=7.5,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.2,
    )
    fig.suptitle(
        f"DIRAC site Green Score comparison — {day} UTC",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.108,
        "CEE = CPU normalization factor × average cores / average TDP     Green Score = CEE / (EI × PUE)     EI = 0.71 × Cscaled + 0.29 × Hscaled",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.19, hspace=0.30)
    return fig


def run(
    max_norm_path: Path,
    minmax_path: Path,
    sites_path: Path,
    config_path: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    parameters = load_site_parameters(sites_path, config_path)
    max_norm = keyed(read_csv(max_norm_path))
    minmax = keyed(read_csv(minmax_path))
    common_keys = sorted(set(max_norm) & set(minmax))

    rows: list[dict[str, Any]] = []
    for timestamp, site in common_keys:
        max_row = max_norm[(timestamp, site)]
        minmax_row = minmax[(timestamp, site)]
        site_parameters = parameters[site]
        ei_max_norm = float(max_row["ei"])
        ei_minmax = float(minmax_row["ei"])
        rows.append(
            {
                "timestamp_utc": timestamp,
                "date_utc": max_row["date_utc"],
                "time_utc": max_row["time_utc"],
                "site": site,
                "wattnet_zone": max_row["wattnet_zone"],
                "pue": site_parameters["pue"],
                "cee": site_parameters["cee"],
                "ei_max_norm": ei_max_norm,
                "green_score_max_norm": green_score(
                    site_parameters["cee"], ei_max_norm, site_parameters["pue"]
                ),
                "ei_minmax": ei_minmax,
                "green_score_minmax": green_score(
                    site_parameters["cee"], ei_minmax, site_parameters["pue"]
                ),
            }
        )

    result_fields = [
        "timestamp_utc", "date_utc", "time_utc", "site", "wattnet_zone",
        "pue", "cee", "ei_max_norm", "green_score_max_norm",
        "ei_minmax", "green_score_minmax",
    ]
    write_csv(output / "dirac_green_scores_15min.csv", rows, result_fields)
    write_csv(
        output / "dirac_site_cee_pue.csv",
        [parameters[site] for site in parameters],
        [
            "site", "avg_tdp_w", "avg_total_cores", "cpu_normalization_factor",
            "pue", "cee",
        ],
    )

    summaries: list[dict[str, Any]] = []
    for site in parameters:
        site_rows = [row for row in rows if row["site"] == site]
        summary: dict[str, Any] = {
            "site": site,
            "wattnet_zone": site_rows[0]["wattnet_zone"],
            "pue": parameters[site]["pue"],
            "cee": parameters[site]["cee"],
            "samples": len(site_rows),
        }
        for field in ("green_score_max_norm", "green_score_minmax"):
            values = np.asarray([float(row[field]) for row in site_rows])
            summary[f"{field}_mean"] = float(values.mean())
            summary[f"{field}_median"] = float(np.median(values))
            summary[f"{field}_min"] = float(values.min())
            summary[f"{field}_max"] = float(values.max())
        summaries.append(summary)
    summaries.sort(key=lambda row: row["green_score_minmax_mean"], reverse=True)
    write_csv(
        output / "dirac_green_score_site_summary.csv",
        summaries,
        list(summaries[0]),
    )

    finite_scores = [
        float(row[field])
        for row in rows
        for field in ("green_score_max_norm", "green_score_minmax")
        if math.isfinite(float(row[field]))
    ]
    y_max = max(finite_scores) * 1.04
    sites = list(parameters)
    days = sorted({row["date_utc"] for row in rows})
    pdf_path = output / "dirac_green_score_daily_comparison.pdf"
    png_paths: list[Path] = []
    with PdfPages(pdf_path) as pdf:
        for day in days:
            figure = plot_day(
                day,
                [row for row in rows if row["date_utc"] == day],
                sites,
                y_max,
            )
            path = output / f"dirac_green_score_comparison_{day}.png"
            figure.savefig(path, dpi=220)
            pdf.savefig(figure)
            plt.close(figure)
            png_paths.append(path)

    report = (
        "DIRAC Green Score comparison\n"
        "CEE = CPU normalization factor * average cores / average TDP\n"
        "EI = 0.71 * Cscaled + 0.29 * Hscaled\n"
        "Green Score = CEE / (EI * PUE)\n"
        "Compared normalization methods: divide by daily maximum; daily min-max.\n"
        "Higher Green Score is preferable; higher EI is worse.\n"
        f"Sites: {len(parameters)}\n"
        f"Comparable site/timestamp rows: {len(rows)}\n"
        "PUE source: configured CIM/KPI /pue endpoint.\n"
        "Warning: CIM reported a location for prague_cesnet_lcg2 differing by more than one degree from sitesnew.csv.\n"
    )
    (output / "README.txt").write_text(report, encoding="utf-8")
    print(report)
    print(pdf_path)
    print(output / "dirac_green_scores_15min.csv")
    print(output / "dirac_green_score_site_summary.csv")
    print(output / "dirac_site_cee_pue.csv")
    for path in png_paths:
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot DIRAC Green Scores for two EI normalizations."
    )
    parser.add_argument("--max-norm", type=Path, default=DEFAULT_MAX_NORM)
    parser.add_argument("--minmax", type=Path, default=DEFAULT_MINMAX)
    parser.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.max_norm, args.minmax, args.sites, args.config, args.output)
