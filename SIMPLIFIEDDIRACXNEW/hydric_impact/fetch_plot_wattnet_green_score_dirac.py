from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.backends.backend_pdf import PdfPages

from fetch_plot_wattnet_week import (
    DASHBOARD_API,
    DIRAC_SITE_ZONES,
    get_dashboard_token,
    iso_z,
)


BASE = Path(__file__).resolve().parent
DEFAULT_SITES = BASE.parent / "sitesnew.csv"
DEFAULT_OUTPUT = (
    BASE
    / "wattnet_last_complete_week_all_zones"
    / "wattnet_green_score_dirac_sites"
)


def parse_day(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), time.min, tzinfo=timezone.utc)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fetch_environmental_score(
    start: datetime, end: datetime, timeout: float
) -> list[dict[str, Any]]:
    with requests.Session() as session:
        token = get_dashboard_token(session, timeout)
        response = session.get(
            f"{DASHBOARD_API}/metrics",
            params={
                "metric": "green-score",
                "scope": "operational",
                "start": iso_z(start),
                "end": iso_z(end),
                "aggregate": "false",
                "use_global": "true",
            },
            headers={"x-dashboard-token": token},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected WattNet EnvironmentalScore response")
    return payload


def flatten_valid_complete(
    payload: list[dict[str, Any]], wanted_zones: set[str]
) -> dict[tuple[str, str], float]:
    points: dict[tuple[str, str], float] = {}
    for zone_record in payload:
        zone = str(zone_record.get("zone", ""))
        if zone not in wanted_zones:
            continue
        for series in zone_record.get("series", []):
            if not bool(series.get("valid")) or series.get("zone_status") != "complete":
                continue
            for timestamp, raw_value in series.get("values", []):
                score = float(raw_value)
                if not 0.0 <= score <= 100.0:
                    raise ValueError(
                        f"WattNet EnvironmentalScore outside [0,100]: zone={zone}, value={score}"
                    )
                points[(zone, str(timestamp))] = score
    return points


def plot_day(
    day: str,
    points: dict[tuple[str, str], float],
    zone_sites: dict[str, list[str]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7.8))
    zones = sorted(zone_sites)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(zones)))
    for zone, color in zip(zones, colors):
        values = sorted(
            (timestamp, score)
            for (point_zone, timestamp), score in points.items()
            if point_zone == zone and timestamp.startswith(day)
        )
        hours = []
        scores = []
        for timestamp, score in values:
            when = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            hours.append(when.hour + when.minute / 60.0)
            scores.append(score)
        sites = zone_sites[zone]
        label = f"{zone}: {sites[0]}" if len(sites) == 1 else f"{zone} ({len(sites)} DIRAC sites)"
        ax.plot(hours, scores, color=color, linewidth=2.1, label=label)

    ax.set_title(
        "EnvironmentalScore returned by WattNet",
        loc="left",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("UTC time")
    ax.set_ylabel("EnvironmentalScore, ES (higher is better)")
    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.grid(True, color="#d8dde1", linewidth=0.7, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.suptitle(
        f"WattNet EnvironmentalScore for DIRAC sites — {day} UTC",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Scope: operational; coverage: global with cross-border flow tracing; valid and complete observations only.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.75, top=0.88, bottom=0.12)
    return fig


def run(
    start: datetime,
    end: datetime,
    sites_path: Path,
    output: Path,
    timeout: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    site_rows = read_csv(sites_path)
    missing = [row["site"] for row in site_rows if row["site"] not in DIRAC_SITE_ZONES]
    if missing:
        raise ValueError(f"No WattNet zone mapping for DIRAC sites: {missing}")

    zone_sites: dict[str, list[str]] = defaultdict(list)
    mapping_rows: list[dict[str, str]] = []
    for row in site_rows:
        zone = DIRAC_SITE_ZONES[row["site"]]
        zone_sites[zone].append(row["site"])
        mapping_rows.append({"site": row["site"], "wattnet_zone": zone})

    payload = fetch_environmental_score(start, end, timeout)
    (output / "wattnet_environmental_score_response.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    points = flatten_valid_complete(payload, set(zone_sites))

    result_rows: list[dict[str, Any]] = []
    for (zone, timestamp), score in sorted(points.items(), key=lambda item: item[0][1]):
        when = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        for site in zone_sites[zone]:
            result_rows.append(
                {
                    "timestamp_utc": iso_z(when),
                    "date_utc": when.date().isoformat(),
                    "time_utc": when.strftime("%H:%M"),
                    "site": site,
                    "wattnet_zone": zone,
                    "environmental_score": score,
                }
            )
    write_csv(
        output / "wattnet_environmental_score_dirac_15min.csv",
        result_rows,
        [
            "timestamp_utc", "date_utc", "time_utc", "site",
            "wattnet_zone", "environmental_score",
        ],
    )
    write_csv(
        output / "dirac_site_zone_mapping.csv",
        mapping_rows,
        ["site", "wattnet_zone"],
    )

    summaries: list[dict[str, Any]] = []
    for site in [row["site"] for row in site_rows]:
        site_values = np.asarray(
            [
                float(row["environmental_score"])
                for row in result_rows
                if row["site"] == site
            ]
        )
        summaries.append(
            {
                "site": site,
                "wattnet_zone": DIRAC_SITE_ZONES[site],
                "samples": int(site_values.size),
                "environmental_score_mean": float(site_values.mean()),
                "environmental_score_median": float(np.median(site_values)),
                "environmental_score_min": float(site_values.min()),
                "environmental_score_max": float(site_values.max()),
            }
        )
    summaries.sort(key=lambda row: row["environmental_score_mean"], reverse=True)
    write_csv(
        output / "wattnet_environmental_score_dirac_summary.csv",
        summaries,
        list(summaries[0]),
    )

    days = sorted({row["date_utc"] for row in result_rows})
    pdf_path = output / "wattnet_environmental_score_dirac_daily_curves.pdf"
    png_paths: list[Path] = []
    with PdfPages(pdf_path) as pdf:
        for day in days:
            figure = plot_day(day, points, zone_sites)
            path = output / f"wattnet_environmental_score_dirac_{day}.png"
            figure.savefig(path, dpi=220)
            pdf.savefig(figure)
            plt.close(figure)
            png_paths.append(path)

    report = (
        "WattNet EnvironmentalScore data for DIRAC sites\n"
        f"Interval: {iso_z(start)} to {iso_z(end)} (end exclusive)\n"
        "Current endpoint metric: green-score (planned name: environmental-score)\n"
        "Scope: operational\n"
        "Coverage: global (cross-border flow tracing enabled)\n"
        "Resolution: 15 minutes; aggregate=false\n"
        "Filtering: valid=true and zone_status=complete\n"
        f"DIRAC sites: {len(site_rows)}\n"
        f"Distinct WattNet zones: {len(zone_sites)}\n"
        f"Site/timestamp rows: {len(result_rows)}\n"
    )
    (output / "README.txt").write_text(report, encoding="utf-8")
    print(report)
    print(pdf_path)
    print(output / "wattnet_environmental_score_dirac_15min.csv")
    print(output / "wattnet_environmental_score_dirac_summary.csv")
    for path in png_paths:
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve WattNet EnvironmentalScore for all DIRAC sites."
    )
    parser.add_argument("--start", default="2026-08-04")
    parser.add_argument("--end", default="2026-08-11")
    parser.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    run(parse_day(args.start), parse_day(args.end), args.sites, args.output, args.timeout)
