from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.backends.backend_pdf import PdfPages


DASHBOARD_API = "https://dashboard.wattnet.eu/api"
BASE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE / "wattnet_last_complete_week_all_zones"
DEFAULT_DIRAC_SITES = BASE.parent / "sitesnew.csv"

# Resolved from sitesnew.csv coordinates against WattNet's official GeoJSON zones.
DIRAC_SITE_ZONES = {
    "SARA-MATRIX": "NL",
    "NIKHEF-ELPROD": "NL",
    "RAL-LCG2": "GB",
    "FZK-LCG2": "DE",
    "UKI-SCOTGRID-GLASGOW": "GB",
    "UKI-SOUTHGRID-RALPP": "GB",
    "IN2P3-IRES": "FR",
    "IN2P3-CPPM": "FR",
    "IN2P3-LPC": "FR",
    "UKI-LT2-IC-HEP": "GB",
    "UKI-NORTHGRID-MAN-HEP": "GB",
    "UKI-NORTHGRID-LANCS-HEP": "GB",
    "TR-03-METU": "TR",
    "GRIF": "FR",
    "NCG-INGRID-PT": "PT",
    "prague_cesnet_lcg2": "CZ",
    "CNR-ILC-PISA": "IT_CNORTH",
}


def parse_utc_day(value: str) -> datetime:
    parsed = date.fromisoformat(value)
    return datetime.combine(parsed, time.min, tzinfo=timezone.utc)


def default_interval(days: int) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return end - timedelta(days=days), end


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def get_dashboard_token(session: requests.Session, timeout: float) -> str:
    response = session.get(f"{DASHBOARD_API}/core", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("token"):
        raise RuntimeError("WattNet dashboard token response has no token")
    return str(payload["token"])


def fetch_metric(
    session: requests.Session,
    token: str,
    metric: str,
    dimension: str,
    start: datetime,
    end: datetime,
    timeout: float,
) -> list[dict[str, Any]]:
    params = {
        "metric": metric,
        "dimension": dimension,
        "scope": "operational",
        "start": iso_z(start),
        "end": iso_z(end),
        "aggregate": "false",
        "use_global": "true",
    }
    response = session.get(
        f"{DASHBOARD_API}/metrics",
        params=params,
        headers={"x-dashboard-token": token},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected WattNet {metric} response")
    return payload


def series_priority(valid: bool, status: str) -> tuple[int, int]:
    return (1 if valid else 0, {"complete": 2, "preview": 1}.get(status, 0))


def flatten_metric(
    payload: list[dict[str, Any]], value_name: str
) -> tuple[dict[tuple[str, datetime], dict[str, Any]], str]:
    points: dict[tuple[str, datetime], dict[str, Any]] = {}
    unit = ""
    for zone_record in payload:
        zone = str(zone_record["zone"])
        unit = unit or str(zone_record.get("unit", ""))
        coverage = str(zone_record.get("coverage", ""))
        for series in zone_record.get("series", []):
            valid = bool(series.get("valid", False))
            status = str(series.get("zone_status", "unknown"))
            priority = series_priority(valid, status)
            for timestamp, raw_value in series.get("values", []):
                when = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                when = when.astimezone(timezone.utc)
                key = (zone, when)
                candidate = {
                    value_name: float(raw_value),
                    f"{value_name}_valid": valid,
                    f"{value_name}_zone_status": status,
                    f"{value_name}_coverage": coverage,
                    "_priority": priority,
                }
                if key not in points or priority > points[key]["_priority"]:
                    points[key] = candidate
    for point in points.values():
        point.pop("_priority", None)
    return points, unit


def combine_points(
    ci: dict[tuple[str, datetime], dict[str, Any]],
    hi: dict[tuple[str, datetime], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for zone, timestamp in sorted(set(ci) | set(hi), key=lambda item: (item[1], item[0])):
        row: dict[str, Any] = {
            "timestamp_utc": iso_z(timestamp),
            "date_utc": timestamp.date().isoformat(),
            "time_utc": timestamp.strftime("%H:%M"),
            "zone": zone,
        }
        row.update(ci.get((zone, timestamp), {}))
        row.update(hi.get((zone, timestamp), {}))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_matrix(
    points: dict[tuple[str, datetime], dict[str, Any]],
    value_name: str,
    zones: list[str],
    day: datetime,
    require_final: bool = False,
) -> np.ndarray:
    result = np.full((len(zones), 96), np.nan, dtype=float)
    zone_index = {zone: index for index, zone in enumerate(zones)}
    day_end = day + timedelta(days=1)
    for (zone, timestamp), fields in points.items():
        if day <= timestamp < day_end and zone in zone_index:
            if require_final and not (
                fields.get(f"{value_name}_valid") is True
                and fields.get(f"{value_name}_zone_status") == "complete"
            ):
                continue
            quarter = timestamp.hour * 4 + timestamp.minute // 15
            result[zone_index[zone], quarter] = float(fields[value_name])
    return result


def finite_limits(values: list[float], logarithmic: bool) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if logarithmic:
        data = data[data > 0]
    if not data.size:
        return (1.0, 10.0) if logarithmic else (0.0, 1.0)
    low = float(data.min()) if logarithmic else 0.0
    high = float(data.max())
    if high <= low:
        high = low * 1.01 if low else 1.0
    return low, high


def daily_figure(
    day: datetime,
    zones: list[str],
    ci_matrix: np.ndarray,
    hi_matrix: np.ndarray,
    ci_limits: tuple[float, float],
    hi_limits: tuple[float, float],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 13), constrained_layout=True)
    cmap_ci = plt.get_cmap("YlOrRd").copy()
    cmap_hi = plt.get_cmap("YlGnBu").copy()
    cmap_ci.set_bad("#d7dce0")
    cmap_hi.set_bad("#d7dce0")

    ci_image = axes[0].imshow(
        np.ma.masked_invalid(ci_matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_ci,
        vmin=ci_limits[0],
        vmax=ci_limits[1],
    )
    hi_image = axes[1].imshow(
        np.ma.masked_invalid(hi_matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_hi,
        vmin=hi_limits[0],
        vmax=hi_limits[1],
    )

    for index, (ax, title) in enumerate(
        zip(axes, ("Carbon intensity (CI)", "AWARE-weighted hydric impact (HI)"))
    ):
        ax.set_title(title, loc="left", fontsize=13, weight="bold")
        ax.set_xlabel("UTC time")
        ax.set_xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "24:00"])
        ax.set_yticks(np.arange(len(zones)), zones if index == 0 else [])
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("WattNet electricity zone" if index == 0 else "")

    fig.colorbar(ci_image, ax=axes[0], pad=0.01, label="gCO2/kWh (operational)")
    fig.colorbar(hi_image, ax=axes[1], pad=0.01, label="stress-L/kWh (operational)")
    fig.suptitle(
        f"WattNet 15-minute environmental intensities — {day.date().isoformat()} UTC",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Global coverage includes cross-border flow tracing. Preview/non-final values are retained and flagged in the CSV; identical scales are used across all days.",
        ha="center",
        fontsize=9,
    )
    return fig


def daily_curve_figure(
    day: datetime,
    zones: list[str],
    ci_matrix: np.ndarray,
    hi_matrix: np.ndarray,
    ci_limits: tuple[float, float],
    hi_limits: tuple[float, float],
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    quarter_hours = np.arange(96, dtype=float) / 4.0
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(zones)))

    for zone_index, (zone, color) in enumerate(zip(zones, colors)):
        axes[0].plot(
            quarter_hours,
            ci_matrix[zone_index],
            color=color,
            linewidth=1.0,
            alpha=0.82,
            label=zone,
        )
        axes[1].plot(
            quarter_hours,
            hi_matrix[zone_index],
            color=color,
            linewidth=1.0,
            alpha=0.82,
            label=zone,
        )

    axes[0].set_title("Carbon intensity (CI)", loc="left", fontsize=13, weight="bold")
    axes[0].set_ylabel("gCO2/kWh (operational)")
    axes[0].set_ylim(ci_limits)
    axes[1].set_title(
        "AWARE-weighted hydric impact (HI)", loc="left", fontsize=13, weight="bold"
    )
    axes[1].set_ylabel("stress-L/kWh (operational)")
    axes[1].set_ylim(hi_limits)
    axes[1].set_xlabel("UTC time")
    axes[1].set_xlim(0.0, 24.0)
    axes[1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])

    for ax in axes:
        ax.grid(True, color="#d8dde1", linewidth=0.65, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=9,
        fontsize=7,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.suptitle(
        f"WattNet 15-minute CI and HI curves — {day.date().isoformat()} UTC",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.105,
        "One curve per WattNet electricity zone; colors are identical in both panels and across all seven days.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.18, hspace=0.28)
    return fig


def daily_scaled_curve_figure(
    day: datetime,
    zones: list[str],
    cscaled: np.ndarray,
    hscaled: np.ndarray,
    ci_max: float,
    hi_max: float,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    quarter_hours = np.arange(96, dtype=float) / 4.0
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(zones)))

    for zone_index, (zone, color) in enumerate(zip(zones, colors)):
        axes[0].plot(
            quarter_hours,
            cscaled[zone_index],
            color=color,
            linewidth=1.0,
            alpha=0.82,
            label=zone,
        )
        axes[1].plot(
            quarter_hours,
            hscaled[zone_index],
            color=color,
            linewidth=1.0,
            alpha=0.82,
            label=zone,
        )

    axes[0].set_title(
        f"Cscaled = CI / CImax   (daily CImax = {ci_max:.2f} gCO2/kWh)",
        loc="left",
        fontsize=13,
        weight="bold",
    )
    axes[0].set_ylabel("Cscaled")
    axes[1].set_title(
        f"Hscaled = HI / HImax   (daily HImax = {hi_max:.2f} stress-L/kWh)",
        loc="left",
        fontsize=13,
        weight="bold",
    )
    axes[1].set_ylabel("Hscaled")
    axes[1].set_xlabel("UTC time")
    axes[1].set_xlim(0.0, 24.0)
    axes[1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])

    for ax in axes:
        ax.set_ylim(0.0, 1.03)
        ax.grid(True, color="#d8dde1", linewidth=0.65, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=9,
        fontsize=7,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.suptitle(
        f"Daily scaled WattNet CI and HI — {day.date().isoformat()} UTC",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.105,
        "One curve per electricity zone. Daily maxima use only valid, complete observations; preview/non-final points are omitted.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.18, hspace=0.28)
    return fig


def load_dirac_site_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sites = [str(row["site"]) for row in rows]
    missing = [site for site in sites if site not in DIRAC_SITE_ZONES]
    if missing:
        raise ValueError(f"DIRAC sites have no WattNet zone mapping: {missing}")
    return [
        {
            "site": site,
            "wattnet_zone": DIRAC_SITE_ZONES[site],
            "latitude": str(row["latitude"]),
            "longitude": str(row["longitude"]),
        }
        for site, row in zip(sites, rows)
    ]


def daily_dirac_ei_figure(
    day: datetime,
    zone_ei: dict[str, np.ndarray],
    zone_sites: dict[str, list[str]],
    ci_max: float,
    hi_max: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7.8))
    quarter_hours = np.arange(96, dtype=float) / 4.0
    zones = sorted(zone_ei)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(zones)))
    for zone, color in zip(zones, colors):
        sites = zone_sites[zone]
        label = f"{zone}: {sites[0]}" if len(sites) == 1 else f"{zone} ({len(sites)} DIRAC sites)"
        ax.plot(
            quarter_hours,
            zone_ei[zone],
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
        f"DIRAC-site environmental burden — {day.date().isoformat()} UTC",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.025,
        f"Daily all-zone references: CImax = {ci_max:.2f} gCO2/kWh; HImax = {hi_max:.2f} stress-L/kWh. Valid, complete observations only.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.73, top=0.88, bottom=0.12)
    return fig


def summary_rows(
    rows: list[dict[str, Any]], zones: list[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for zone in zones:
        zone_rows = [row for row in rows if row["zone"] == zone]
        entry: dict[str, Any] = {"zone": zone}
        for value_name in ("ci_gco2_per_kwh", "hi_stress_l_per_kwh"):
            values = np.asarray(
                [float(row[value_name]) for row in zone_rows if row.get(value_name) not in (None, "")],
                dtype=float,
            )
            entry[f"{value_name}_samples"] = int(values.size)
            entry[f"{value_name}_mean"] = float(values.mean()) if values.size else ""
            entry[f"{value_name}_median"] = float(np.median(values)) if values.size else ""
            entry[f"{value_name}_min"] = float(values.min()) if values.size else ""
            entry[f"{value_name}_max"] = float(values.max()) if values.size else ""
        output.append(entry)
    return output


def run(start: datetime, end: datetime, output_dir: Path, timeout: float) -> None:
    if end <= start or start.time() != time.min or end.time() != time.min:
        raise ValueError("Start and end must be distinct UTC day boundaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        token = get_dashboard_token(session, timeout)
        carbon_payload = fetch_metric(
            session, token, "footprint", "carbon", start, end, timeout
        )
        hydric_payload = fetch_metric(
            session, token, "impact", "water", start, end, timeout
        )

    (output_dir / "wattnet_ci_response.json").write_text(
        json.dumps(carbon_payload, indent=2), encoding="utf-8"
    )
    (output_dir / "wattnet_hi_response.json").write_text(
        json.dumps(hydric_payload, indent=2), encoding="utf-8"
    )

    ci, ci_unit = flatten_metric(carbon_payload, "ci_gco2_per_kwh")
    hi, hi_unit = flatten_metric(hydric_payload, "hi_stress_l_per_kwh")
    rows = combine_points(ci, hi)
    zones = sorted({zone for zone, _ in ci} | {zone for zone, _ in hi})

    fields = [
        "timestamp_utc", "date_utc", "time_utc", "zone",
        "ci_gco2_per_kwh", "ci_gco2_per_kwh_valid", "ci_gco2_per_kwh_zone_status", "ci_gco2_per_kwh_coverage",
        "hi_stress_l_per_kwh", "hi_stress_l_per_kwh_valid", "hi_stress_l_per_kwh_zone_status", "hi_stress_l_per_kwh_coverage",
    ]
    write_csv(output_dir / "wattnet_ci_hi_15min.csv", rows, fields)

    summaries = summary_rows(rows, zones)
    summary_fields = list(summaries[0]) if summaries else ["zone"]
    write_csv(output_dir / "wattnet_ci_hi_zone_summary.csv", summaries, summary_fields)

    ci_limits = finite_limits([point["ci_gco2_per_kwh"] for point in ci.values()], False)
    hi_limits = finite_limits([point["hi_stress_l_per_kwh"] for point in hi.values()], False)
    pdf_path = output_dir / "wattnet_ci_hi_daily_heatmaps.pdf"
    plot_paths: list[Path] = []
    with PdfPages(pdf_path) as pdf:
        day = start
        while day < end:
            figure = daily_figure(
                day,
                zones,
                make_matrix(ci, "ci_gco2_per_kwh", zones, day),
                make_matrix(hi, "hi_stress_l_per_kwh", zones, day),
                ci_limits,
                hi_limits,
            )
            path = output_dir / f"wattnet_ci_hi_{day.date().isoformat()}.png"
            figure.savefig(path, dpi=220, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            plot_paths.append(path)
            day += timedelta(days=1)

    curve_pdf_path = output_dir / "wattnet_ci_hi_daily_curves.pdf"
    curve_paths: list[Path] = []
    with PdfPages(curve_pdf_path) as pdf:
        day = start
        while day < end:
            figure = daily_curve_figure(
                day,
                zones,
                make_matrix(ci, "ci_gco2_per_kwh", zones, day),
                make_matrix(hi, "hi_stress_l_per_kwh", zones, day),
                ci_limits,
                hi_limits,
            )
            path = output_dir / f"wattnet_ci_hi_curves_{day.date().isoformat()}.png"
            figure.savefig(path, dpi=220)
            pdf.savefig(figure)
            plt.close(figure)
            curve_paths.append(path)
            day += timedelta(days=1)

    scaled_pdf_path = output_dir / "wattnet_cscaled_hscaled_daily_curves.pdf"
    scaled_paths: list[Path] = []
    daily_references: list[dict[str, Any]] = []
    with PdfPages(scaled_pdf_path) as pdf:
        day = start
        while day < end:
            ci_final = make_matrix(
                ci, "ci_gco2_per_kwh", zones, day, require_final=True
            )
            hi_final = make_matrix(
                hi, "hi_stress_l_per_kwh", zones, day, require_final=True
            )
            ci_max = float(np.nanmax(ci_final))
            hi_max = float(np.nanmax(hi_final))
            ci_min = float(np.nanmin(ci_final))
            hi_min = float(np.nanmin(hi_final))
            cscaled = ci_final / ci_max
            hscaled = hi_final / hi_max
            daily_references.append(
                {
                    "date_utc": day.date().isoformat(),
                    "ci_min_gco2_per_kwh": ci_min,
                    "ci_max_gco2_per_kwh": ci_max,
                    "hi_min_stress_l_per_kwh": hi_min,
                    "hi_max_stress_l_per_kwh": hi_max,
                }
            )
            figure = daily_scaled_curve_figure(
                day, zones, cscaled, hscaled, ci_max, hi_max
            )
            path = output_dir / f"wattnet_cscaled_hscaled_{day.date().isoformat()}.png"
            figure.savefig(path, dpi=220)
            pdf.savefig(figure)
            plt.close(figure)
            scaled_paths.append(path)
            day += timedelta(days=1)

    write_csv(
        output_dir / "wattnet_daily_scaling_references.csv",
        daily_references,
        list(daily_references[0]) if daily_references else ["date_utc"],
    )

    site_mapping = load_dirac_site_mapping(DEFAULT_DIRAC_SITES)
    write_csv(
        output_dir / "wattnet_dirac_site_zone_mapping.csv",
        site_mapping,
        ["site", "wattnet_zone", "latitude", "longitude"],
    )
    zone_sites: dict[str, list[str]] = {}
    for mapping in site_mapping:
        zone_sites.setdefault(mapping["wattnet_zone"], []).append(mapping["site"])

    dirac_ei_rows: list[dict[str, Any]] = []
    dirac_ei_paths: list[Path] = []
    dirac_ei_pdf_path = output_dir / "wattnet_dirac_ei_daily_curves.pdf"
    zone_index = {zone: index for index, zone in enumerate(zones)}
    with PdfPages(dirac_ei_pdf_path) as pdf:
        day = start
        while day < end:
            ci_final = make_matrix(
                ci, "ci_gco2_per_kwh", zones, day, require_final=True
            )
            hi_final = make_matrix(
                hi, "hi_stress_l_per_kwh", zones, day, require_final=True
            )
            ci_max = float(np.nanmax(ci_final))
            hi_max = float(np.nanmax(hi_final))
            zone_ei: dict[str, np.ndarray] = {}
            for zone in sorted(zone_sites):
                index = zone_index[zone]
                cscaled = ci_final[index] / ci_max
                hscaled = hi_final[index] / hi_max
                ei = 0.71 * cscaled + 0.29 * hscaled
                zone_ei[zone] = ei
                for quarter, value in enumerate(ei):
                    if not np.isfinite(value):
                        continue
                    timestamp = day + timedelta(minutes=15 * quarter)
                    for site in zone_sites[zone]:
                        dirac_ei_rows.append(
                            {
                                "timestamp_utc": iso_z(timestamp),
                                "date_utc": day.date().isoformat(),
                                "time_utc": timestamp.strftime("%H:%M"),
                                "site": site,
                                "wattnet_zone": zone,
                                "ci_max_gco2_per_kwh": ci_max,
                                "hi_max_stress_l_per_kwh": hi_max,
                                "cscaled": float(cscaled[quarter]),
                                "hscaled": float(hscaled[quarter]),
                                "ei": float(value),
                            }
                        )
            figure = daily_dirac_ei_figure(
                day, zone_ei, zone_sites, ci_max, hi_max
            )
            path = output_dir / f"wattnet_dirac_ei_{day.date().isoformat()}.png"
            figure.savefig(path, dpi=220)
            pdf.savefig(figure)
            plt.close(figure)
            dirac_ei_paths.append(path)
            day += timedelta(days=1)

    write_csv(
        output_dir / "wattnet_dirac_ei_15min.csv",
        dirac_ei_rows,
        [
            "timestamp_utc", "date_utc", "time_utc", "site", "wattnet_zone",
            "ci_max_gco2_per_kwh", "hi_max_stress_l_per_kwh",
            "cscaled", "hscaled", "ei",
        ],
    )

    expected = len(zones) * int((end - start).total_seconds() // 900)
    ci_status = Counter(point["ci_gco2_per_kwh_zone_status"] for point in ci.values())
    hi_status = Counter(point["hi_stress_l_per_kwh_zone_status"] for point in hi.values())
    ci_valid = Counter(bool(point["ci_gco2_per_kwh_valid"]) for point in ci.values())
    hi_valid = Counter(bool(point["hi_stress_l_per_kwh_valid"]) for point in hi.values())
    report = (
        "WattNet all-zone 15-minute CI and HI dataset\n"
        f"Interval: {iso_z(start)} to {iso_z(end)} (end exclusive)\n"
        f"Days: {(end - start).days}\n"
        f"Zones with data: {len(zones)}\n"
        f"Expected samples per metric for a complete grid: {expected}\n"
        f"CI samples: {len(ci)}; unit reported by API: {ci_unit}\n"
        f"HI samples: {len(hi)}; unit reported by API: {hi_unit}\n"
        f"CI zone-status counts: {dict(ci_status)}\n"
        f"HI zone-status counts: {dict(hi_status)}\n"
        f"CI validity counts: {dict(ci_valid)}\n"
        f"HI validity counts: {dict(hi_valid)}\n"
        "Scope: operational\n"
        "Coverage: global (cross-border flow tracing enabled)\n"
        "Resolution requested: native 15 minutes; aggregate=false\n"
        f"Zones: {', '.join(zones)}\n"
    )
    (output_dir / "README.txt").write_text(report, encoding="utf-8")
    print(report)
    print(output_dir / "wattnet_ci_hi_15min.csv")
    print(output_dir / "wattnet_ci_hi_zone_summary.csv")
    print(pdf_path)
    print(curve_pdf_path)
    print(scaled_pdf_path)
    print(dirac_ei_pdf_path)
    for path in plot_paths:
        print(path)
    for path in curve_paths:
        print(path)
    for path in scaled_paths:
        print(path)
    for path in dirac_ei_paths:
        print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and plot 15-minute WattNet CI and HI for all zones."
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--start", help="UTC start day, YYYY-MM-DD")
    parser.add_argument("--end", help="Exclusive UTC end day, YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if bool(args.start) != bool(args.end):
        raise SystemExit("--start and --end must be supplied together")
    if args.start:
        interval_start, interval_end = parse_utc_day(args.start), parse_utc_day(args.end)
    else:
        interval_start, interval_end = default_interval(args.days)
    run(interval_start, interval_end, args.output_dir, args.timeout)
