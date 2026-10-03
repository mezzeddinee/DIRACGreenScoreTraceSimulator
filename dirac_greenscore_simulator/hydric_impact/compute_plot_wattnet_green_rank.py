from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


BASE = Path(__file__).resolve().parent
WEEK = BASE / "wattnet_last_complete_week_all_zones"
_NEW_ENVIRONMENTAL_SCORE = (
    WEEK
    / "wattnet_green_score_dirac_sites"
    / "wattnet_environmental_score_dirac_15min.csv"
)
_LEGACY_ENVIRONMENTAL_SCORE = (
    WEEK / "wattnet_green_score_dirac_sites" / "wattnet_green_score_dirac_15min.csv"
)
DEFAULT_ENVIRONMENTAL_SCORE = (
    _NEW_ENVIRONMENTAL_SCORE
    if _NEW_ENVIRONMENTAL_SCORE.exists()
    else _LEGACY_ENVIRONMENTAL_SCORE
)
DEFAULT_SITE_PARAMETERS = (
    WEEK / "dirac_green_score_comparison" / "dirac_site_cee_pue.csv"
)
DEFAULT_OUTPUT = WEEK / "wattnet_green_score_dirac_sites" / "final_green_rank"


def green_score_from_environmental_score(
    environmental_score: float, cee: float, pue: float
) -> tuple[float, float]:
    """Return (ES-derived burden, final DIRAC GreenScore)."""
    if not 0.0 <= environmental_score <= 100.0:
        raise ValueError("EnvironmentalScore must be in [0, 100]")
    if cee <= 0.0 or pue <= 0.0:
        raise ValueError("CEE and PUE must be positive")
    burden = 1.0 - environmental_score / 100.0
    if burden <= 0.0:
        raise ValueError("EnvironmentalScore=100 gives an undefined GreenScore")
    return burden, cee / (pue * burden)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_day(
    day: str,
    rows: list[dict[str, Any]],
    sites: list[str],
    y_max: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 9))
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
        ax.plot(
            hours,
            [row["green_score"] for row in site_rows],
            color=color,
            linewidth=1.75,
            label=site,
        )

    ax.set_title(
        "GreenScore = CEE / [PUE × (1 − ES/100)]",
        loc="left",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("UTC time")
    ax.set_ylabel("GreenScore (higher is better)")
    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.grid(True, color="#d8dde1", linewidth=0.7, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=5,
        fontsize=7.5,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.2,
    )
    fig.suptitle(
        f"Final ES-based DIRAC GreenScore — {day} UTC",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.045,
        "WattNet ES is converted to proportional burden as 1 − ES/100 before combining it with CEE and PUE.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.25)
    return fig


def run(environmental_score_path: Path, parameters_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scores = read_csv(environmental_score_path)
    parameter_rows = read_csv(parameters_path)
    parameters = {row["site"]: row for row in parameter_rows}
    missing = sorted({row["site"] for row in scores} - set(parameters))
    if missing:
        raise ValueError(f"Missing CEE/PUE parameters for sites: {missing}")

    rows: list[dict[str, Any]] = []
    for row in scores:
        site = row["site"]
        cee = float(parameters[site]["cee"])
        pue = float(parameters[site]["pue"])
        raw_score = row.get("environmental_score") or row.get("wattnet_green_score")
        if raw_score is None:
            raise ValueError("Input must contain environmental_score")
        environmental_score = float(raw_score)
        environmental_burden, green_score = green_score_from_environmental_score(
            environmental_score, cee, pue
        )
        rows.append(
            {
                "timestamp_utc": row["timestamp_utc"],
                "date_utc": row["date_utc"],
                "time_utc": row["time_utc"],
                "site": site,
                "wattnet_zone": row["wattnet_zone"],
                "environmental_score": environmental_score,
                "environmental_burden": environmental_burden,
                "cee": cee,
                "pue": pue,
                "green_score": green_score,
            }
        )
    write_csv(
        output / "dirac_environmental_score_to_green_score_15min.csv",
        rows,
        [
            "timestamp_utc", "date_utc", "time_utc", "site", "wattnet_zone",
            "environmental_score", "environmental_burden", "cee", "pue",
            "green_score",
        ],
    )

    sites = [row["site"] for row in parameter_rows]
    summaries: list[dict[str, Any]] = []
    for site in sites:
        site_rows = [row for row in rows if row["site"] == site]
        values = np.asarray([float(row["green_score"]) for row in site_rows])
        summaries.append(
            {
                "site": site,
                "wattnet_zone": site_rows[0]["wattnet_zone"],
                "pue": float(parameters[site]["pue"]),
                "cee": float(parameters[site]["cee"]),
                "samples": int(values.size),
                "green_score_mean": float(values.mean()),
                "green_score_median": float(np.median(values)),
                "green_score_min": float(values.min()),
                "green_score_max": float(values.max()),
            }
        )
    summaries.sort(key=lambda row: row["green_score_mean"], reverse=True)
    write_csv(
        output / "dirac_es_based_green_score_site_summary.csv",
        summaries,
        list(summaries[0]),
    )

    y_max = max(float(row["green_score"]) for row in rows) * 1.04
    days = sorted({row["date_utc"] for row in rows})
    pdf_path = output / "dirac_es_based_green_score_daily_curves.pdf"
    png_paths: list[Path] = []
    with PdfPages(pdf_path) as pdf:
        for day in days:
            figure = plot_day(
                day,
                [row for row in rows if row["date_utc"] == day],
                sites,
                y_max,
            )
            path = output / f"dirac_es_based_green_score_{day}.png"
            figure.savefig(path, dpi=220)
            pdf.savefig(figure)
            plt.close(figure)
            png_paths.append(path)

    report = (
        "Final ES-based DIRAC GreenScore\n"
        "Formula: GreenScore = CEE / [PUE * (1 - ES/100)]\n"
        "WattNet EnvironmentalScore scale: 0--100; higher is better\n"
        "CEE = CPU normalization factor * average cores / average TDP\n"
        "Higher GreenScore is preferable.\n"
        f"Sites: {len(sites)}\n"
        f"Site/timestamp rows: {len(rows)}\n"
    )
    (output / "README.txt").write_text(report, encoding="utf-8")
    print(report)
    print(pdf_path)
    print(output / "dirac_environmental_score_to_green_score_15min.csv")
    print(output / "dirac_es_based_green_score_site_summary.csv")
    for path in png_paths:
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute final DIRAC GreenScore from WattNet EnvironmentalScore."
    )
    parser.add_argument(
        "--environmental-score", "--green-score",
        dest="environmental_score",
        type=Path,
        default=DEFAULT_ENVIRONMENTAL_SCORE,
    )
    parser.add_argument("--site-parameters", type=Path, default=DEFAULT_SITE_PARAMETERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.environmental_score, args.site_parameters, args.output)
