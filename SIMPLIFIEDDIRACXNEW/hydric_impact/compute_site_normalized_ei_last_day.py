from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
WEEK = BASE / "wattnet_last_complete_week_all_zones"
DATA = WEEK / "wattnet_ci_hi_15min.csv"
MAPPING = WEEK / "wattnet_dirac_site_zone_mapping.csv"
OUTPUT = WEEK / "site_normalized_ei_last_day"
SITE_PARAMETERS = WEEK / "dirac_green_score_comparison" / "dirac_site_cee_pue.csv"

CARBON_WEIGHT = 0.71
HYDRIC_WEIGHT = 0.29


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scale(value: float, minimum: float, maximum: float) -> float:
    width = maximum - minimum
    return (value - minimum) / width if width else 0.0


def main() -> None:
    source_rows = read_rows(DATA)
    mappings = read_rows(MAPPING)
    last_day = max(row["date_utc"] for row in source_rows)

    by_zone: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_zone[row["zone"]].append(row)

    detail: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for mapping in mappings:
        site = mapping["site"]
        zone = mapping["wattnet_zone"]
        reference = by_zone[zone]
        ci_values = [float(row["ci_gco2_per_kwh"]) for row in reference]
        hi_values = [float(row["hi_stress_l_per_kwh"]) for row in reference]
        ci_min, ci_max = min(ci_values), max(ci_values)
        hi_min, hi_max = min(hi_values), max(hi_values)

        site_detail: list[dict[str, object]] = []
        for row in reference:
            if row["date_utc"] != last_day:
                continue
            ci = float(row["ci_gco2_per_kwh"])
            hi = float(row["hi_stress_l_per_kwh"])
            cscaled = scale(ci, ci_min, ci_max)
            hscaled = scale(hi, hi_min, hi_max)
            ei = CARBON_WEIGHT * cscaled + HYDRIC_WEIGHT * hscaled
            site_detail.append(
                {
                    "timestamp_utc": row["timestamp_utc"],
                    "date_utc": row["date_utc"],
                    "time_utc": row["time_utc"],
                    "site": site,
                    "wattnet_zone": zone,
                    "ci": ci,
                    "hi": hi,
                    "ci_site_week_min": ci_min,
                    "ci_site_week_max": ci_max,
                    "hi_site_week_min": hi_min,
                    "hi_site_week_max": hi_max,
                    "cscaled": cscaled,
                    "hscaled": hscaled,
                    "ei": ei,
                }
            )

        detail.extend(site_detail)
        summary.append(
            {
                "date_utc": last_day,
                "site": site,
                "wattnet_zone": zone,
                "intervals": len(site_detail),
                "ci_site_week_min": ci_min,
                "ci_site_week_max": ci_max,
                "hi_site_week_min": hi_min,
                "hi_site_week_max": hi_max,
                "mean_cscaled": sum(float(row["cscaled"]) for row in site_detail)
                / len(site_detail),
                "mean_hscaled": sum(float(row["hscaled"]) for row in site_detail)
                / len(site_detail),
                "mean_ei": sum(float(row["ei"]) for row in site_detail)
                / len(site_detail),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    detail_fields = list(detail[0])
    with (OUTPUT / "site_ei_15min_last_day.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail)

    summary.sort(key=lambda row: float(row["mean_ei"]))
    summary_fields = list(summary[0])
    with (OUTPUT / "site_average_ei_last_day.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    parameters = {row["site"]: row for row in read_rows(SITE_PARAMETERS)}
    rankings: list[dict[str, object]] = []
    for row in summary:
        site = str(row["site"])
        cee = float(parameters[site]["cee"])
        pue = float(parameters[site]["pue"])
        mean_ei = float(row["mean_ei"])
        rankings.append(
            {
                "date_utc": last_day,
                "site": site,
                "wattnet_zone": row["wattnet_zone"],
                "cee": cee,
                "mean_ei": mean_ei,
                "pue": pue,
                "gsrank": cee / (mean_ei * pue),
            }
        )
    rankings.sort(key=lambda row: float(row["gsrank"]), reverse=True)
    with (OUTPUT / "site_gsrank_last_day.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rankings[0]))
        writer.writeheader()
        writer.writerows(rankings)

    maxnorm_detail: list[dict[str, object]] = []
    maxnorm_summary: list[dict[str, object]] = []
    for mapping in mappings:
        site = mapping["site"]
        zone = mapping["wattnet_zone"]
        reference = by_zone[zone]
        ci_max = max(float(row["ci_gco2_per_kwh"]) for row in reference)
        hi_max = max(float(row["hi_stress_l_per_kwh"]) for row in reference)
        site_detail: list[dict[str, object]] = []
        for row in reference:
            if row["date_utc"] != last_day:
                continue
            ci = float(row["ci_gco2_per_kwh"])
            hi = float(row["hi_stress_l_per_kwh"])
            cscaled = ci / ci_max if ci_max else 0.0
            hscaled = hi / hi_max if hi_max else 0.0
            ei = CARBON_WEIGHT * cscaled + HYDRIC_WEIGHT * hscaled
            site_detail.append(
                {
                    "timestamp_utc": row["timestamp_utc"],
                    "date_utc": row["date_utc"],
                    "time_utc": row["time_utc"],
                    "site": site,
                    "wattnet_zone": zone,
                    "ci": ci,
                    "hi": hi,
                    "ci_site_week_max": ci_max,
                    "hi_site_week_max": hi_max,
                    "cscaled": cscaled,
                    "hscaled": hscaled,
                    "ei": ei,
                }
            )
        maxnorm_detail.extend(site_detail)
        mean_cscaled = sum(float(row["cscaled"]) for row in site_detail) / len(
            site_detail
        )
        mean_hscaled = sum(float(row["hscaled"]) for row in site_detail) / len(
            site_detail
        )
        mean_ei = sum(float(row["ei"]) for row in site_detail) / len(site_detail)
        cee = float(parameters[site]["cee"])
        pue = float(parameters[site]["pue"])
        maxnorm_summary.append(
            {
                "date_utc": last_day,
                "site": site,
                "wattnet_zone": zone,
                "intervals": len(site_detail),
                "ci_site_week_max": ci_max,
                "hi_site_week_max": hi_max,
                "mean_cscaled": mean_cscaled,
                "mean_hscaled": mean_hscaled,
                "mean_ei": mean_ei,
                "cee": cee,
                "pue": pue,
                "gsrank": cee / (mean_ei * pue),
            }
        )

    with (OUTPUT / "site_maxnorm_ei_15min_last_day.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(maxnorm_detail[0]))
        writer.writeheader()
        writer.writerows(maxnorm_detail)

    maxnorm_summary.sort(key=lambda row: float(row["gsrank"]), reverse=True)
    with (OUTPUT / "site_maxnorm_summary_and_gsrank_last_day.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(maxnorm_summary[0]))
        writer.writeheader()
        writer.writerows(maxnorm_summary)

    # Global normalization references: every WattNet zone and every timestamp
    # in the seven-day source dataset contributes to the four extrema.
    all_ci = [float(row["ci_gco2_per_kwh"]) for row in source_rows]
    all_hi = [float(row["hi_stress_l_per_kwh"]) for row in source_rows]
    global_ci_min, global_ci_max = min(all_ci), max(all_ci)
    global_hi_min, global_hi_max = min(all_hi), max(all_hi)
    global_detail: list[dict[str, object]] = []
    global_summary: list[dict[str, object]] = []
    for mapping in mappings:
        site = mapping["site"]
        zone = mapping["wattnet_zone"]
        site_detail: list[dict[str, object]] = []
        for row in by_zone[zone]:
            if row["date_utc"] != last_day:
                continue
            ci = float(row["ci_gco2_per_kwh"])
            hi = float(row["hi_stress_l_per_kwh"])
            cscaled_max = ci / global_ci_max if global_ci_max else 0.0
            hscaled_max = hi / global_hi_max if global_hi_max else 0.0
            cscaled_minmax = scale(ci, global_ci_min, global_ci_max)
            hscaled_minmax = scale(hi, global_hi_min, global_hi_max)
            ei_max = CARBON_WEIGHT * cscaled_max + HYDRIC_WEIGHT * hscaled_max
            ei_minmax = (
                CARBON_WEIGHT * cscaled_minmax
                + HYDRIC_WEIGHT * hscaled_minmax
            )
            site_detail.append(
                {
                    "timestamp_utc": row["timestamp_utc"],
                    "date_utc": row["date_utc"],
                    "time_utc": row["time_utc"],
                    "site": site,
                    "wattnet_zone": zone,
                    "ci": ci,
                    "hi": hi,
                    "global_ci_min": global_ci_min,
                    "global_ci_max": global_ci_max,
                    "global_hi_min": global_hi_min,
                    "global_hi_max": global_hi_max,
                    "cscaled_maxnorm": cscaled_max,
                    "hscaled_maxnorm": hscaled_max,
                    "ei_maxnorm": ei_max,
                    "cscaled_minmax": cscaled_minmax,
                    "hscaled_minmax": hscaled_minmax,
                    "ei_minmax": ei_minmax,
                }
            )
        global_detail.extend(site_detail)
        mean_ei_max = sum(float(row["ei_maxnorm"]) for row in site_detail) / len(
            site_detail
        )
        mean_ei_minmax = sum(
            float(row["ei_minmax"]) for row in site_detail
        ) / len(site_detail)
        cee = float(parameters[site]["cee"])
        pue = float(parameters[site]["pue"])
        global_summary.append(
            {
                "date_utc": last_day,
                "site": site,
                "wattnet_zone": zone,
                "intervals": len(site_detail),
                "global_ci_min": global_ci_min,
                "global_ci_max": global_ci_max,
                "global_hi_min": global_hi_min,
                "global_hi_max": global_hi_max,
                "mean_ei_maxnorm": mean_ei_max,
                "gsrank_maxnorm": cee / (mean_ei_max * pue),
                "mean_ei_minmax": mean_ei_minmax,
                "gsrank_minmax": cee / (mean_ei_minmax * pue),
                "cee": cee,
                "pue": pue,
            }
        )

    with (OUTPUT / "global_bounds_both_normalizations_ei_15min_last_day.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(global_detail[0]))
        writer.writeheader()
        writer.writerows(global_detail)

    global_summary.sort(
        key=lambda row: float(row["gsrank_minmax"]), reverse=True
    )
    with (
        OUTPUT / "global_bounds_both_normalizations_summary_gsrank_last_day.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(global_summary[0]))
        writer.writeheader()
        writer.writerows(global_summary)

    print(f"Last complete day: {last_day}")
    print(
        f"Wrote {len(detail)} interval rows, {len(summary)} site summaries, "
        f"and {len(rankings)} site rankings; site- and global-reference results written"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
