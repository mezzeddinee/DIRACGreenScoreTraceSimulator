#!/usr/bin/env python3
"""Build an A0 poster comparing boundary-trimmed March--June DIRAC runs."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SIM = ROOT / "dirac_greenscore_simulator"
RUN_ROOT = SIM / "hydric_impact" / "timeseries"
OUT = ROOT / "overleaf_dirac_three_month_boundary_trimmed_A0_poster_2026-09-03"
FIGURES = OUT / "figures"
DATA = OUT / "data"
TEMPLATE = ROOT / "overleaf_three_month_boundary_trimmed_A0_poster_main.tex"


@dataclass(frozen=True)
class MonthSpec:
    key: str
    label: str
    trace: Path
    random_run: Path
    green_run: Path
    full_start: pd.Timestamp
    full_end: pd.Timestamp
    analysis_start: pd.Timestamp
    analysis_end: pd.Timestamp
    expected_full_jobs: int


MONTHS = (
    MonthSpec(
        key="march",
        label="March",
        trace=ROOT / "trace_2026_03_01_to_07.csv",
        random_run=RUN_ROOT / "trace_march_7d_random_rerun" / "completed_jobs.csv",
        green_run=RUN_ROOT / "trace_march_7d_greenscore_rerun" / "completed_jobs.csv",
        full_start=pd.Timestamp("2026-03-01 00:00:00"),
        full_end=pd.Timestamp("2026-03-08 00:00:00"),
        analysis_start=pd.Timestamp("2026-03-02 00:00:00"),
        analysis_end=pd.Timestamp("2026-03-07 00:00:00"),
        expected_full_jobs=727_014,
    ),
    MonthSpec(
        key="april",
        label="April",
        trace=ROOT / "trace_2026_04_01_to_07.csv",
        random_run=RUN_ROOT / "trace_april_7d_random_rerun" / "completed_jobs.csv",
        green_run=RUN_ROOT / "trace_april_7d_greenscore_rerun" / "completed_jobs.csv",
        full_start=pd.Timestamp("2026-04-01 00:00:00"),
        full_end=pd.Timestamp("2026-04-08 00:00:00"),
        analysis_start=pd.Timestamp("2026-04-02 00:00:00"),
        analysis_end=pd.Timestamp("2026-04-07 00:00:00"),
        expected_full_jobs=700_514,
    ),
    MonthSpec(
        key="may",
        label="May",
        trace=ROOT / "trace_2026_05_01_to_07.csv",
        random_run=RUN_ROOT / "trace_may_7d_random_rerun" / "completed_jobs.csv",
        green_run=RUN_ROOT / "trace_may_7d_greenscore_rerun" / "completed_jobs.csv",
        full_start=pd.Timestamp("2026-05-01 00:00:00"),
        full_end=pd.Timestamp("2026-05-08 00:00:00"),
        analysis_start=pd.Timestamp("2026-05-02 00:00:00"),
        analysis_end=pd.Timestamp("2026-05-07 00:00:00"),
        expected_full_jobs=341_275,
    ),
    MonthSpec(
        key="june",
        label="June",
        trace=ROOT / "trace_2026_06_01_to_07.csv",
        random_run=RUN_ROOT / "trace_7d_random_rerun" / "completed_jobs.csv",
        green_run=RUN_ROOT / "trace_7d_greenscore_rerun" / "completed_jobs.csv",
        full_start=pd.Timestamp("2026-06-01 00:00:00"),
        full_end=pd.Timestamp("2026-06-08 00:00:00"),
        analysis_start=pd.Timestamp("2026-06-02 00:00:00"),
        analysis_end=pd.Timestamp("2026-06-07 00:00:00"),
        expected_full_jobs=997_370,
    ),
)

SITE_ORDER = ["IN2P3-IRES", "SARA-MATRIX", "NCG-INGRID-PT", "TR-03-METU"]
COUNTRIES = {
    "IN2P3-IRES": "France",
    "SARA-MATRIX": "Netherlands",
    "NCG-INGRID-PT": "Portugal",
    "TR-03-METU": "Turkey",
}
POLICIES = ("Randomized", "GreenScore-based")
POLICY_COLORS = {"Randomized": "#777777", "GreenScore-based": "#228B5A"}
SITE_COLORS = {
    "IN2P3-IRES": "#2166AC",
    "SARA-MATRIX": "#67A9CF",
    "NCG-INGRID-PT": "#FDAE61",
    "TR-03-METU": "#D73027",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.useoffset": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)


def read_site_configuration() -> dict[str, dict[str, float | int | str]]:
    with (SIM / "sites.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if {row["site"] for row in rows} != set(SITE_ORDER):
        raise ValueError("Active sites.csv differs from the expected four-site set")
    ranked = sorted(rows, key=lambda row: float(row["greenhydric"]), reverse=True)
    ranks = {row["site"]: index + 1 for index, row in enumerate(ranked)}
    return {
        row["site"]: {
            "country": COUNTRIES[row["site"]],
            "rank": ranks[row["site"]],
            "score": float(row["greenhydric"]),
            "pue": float(row["pue"]),
        }
        for row in rows
    }


def read_trace(
    spec: MonthSpec,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, float | int | pd.Timestamp]]:
    required = [
        "job_id",
        "submit_time",
        "runtime_min",
        "norm_cpu_seconds",
        "cores_used",
    ]
    if not spec.trace.exists():
        raise FileNotFoundError(spec.trace)
    trace = pd.read_csv(spec.trace, usecols=required)
    trace["submit_time"] = pd.to_datetime(trace["submit_time"], errors="raise")
    if len(trace) != spec.expected_full_jobs:
        raise ValueError(f"{spec.label}: unexpected trace row count {len(trace)}")
    if trace["job_id"].nunique() != len(trace):
        raise ValueError(f"{spec.label}: duplicate trace job IDs")
    if set(trace["cores_used"].astype(int).unique()) != {1}:
        raise ValueError(f"{spec.label}: workload is not exclusively single-core")
    if trace["submit_time"].min() < spec.full_start or trace["submit_time"].max() >= spec.full_end:
        raise ValueError(f"{spec.label}: trace submissions fall outside the seven-day window")

    retained = trace[
        (trace["submit_time"] >= spec.analysis_start)
        & (trace["submit_time"] < spec.analysis_end)
    ].copy()
    if retained.empty:
        raise ValueError(f"{spec.label}: empty retained cohort")
    retained_work = retained.set_index("job_id")["norm_cpu_seconds"].sort_index()

    minute_index = pd.date_range(
        spec.analysis_start, spec.analysis_end - pd.Timedelta(minutes=1), freq="min"
    )
    minute = (
        retained.set_index("submit_time")
        .resample("1min")
        .agg(
            jobs=("job_id", "size"),
            runtime_min=("runtime_min", "sum"),
            norm_cpu_seconds=("norm_cpu_seconds", "sum"),
        )
        .reindex(minute_index, fill_value=0)
    )
    moving = minute.rolling(15, center=True, min_periods=1).mean()
    moving.columns = [f"{column}_moving_average_15min" for column in moving.columns]
    profile = minute.join(moving).reset_index(names="timestamp")
    profile.insert(0, "month", spec.label)
    profile["elapsed_hours"] = (
        profile["timestamp"] - spec.analysis_start
    ).dt.total_seconds() / 3600.0
    profile["elapsed_days"] = profile["elapsed_hours"] / 24.0

    daily = (
        retained.assign(
            retained_day=(retained["submit_time"].dt.floor("D") - spec.analysis_start).dt.days + 1
        )
        .groupby("retained_day")
        .agg(
            jobs=("job_id", "size"),
            runtime_min=("runtime_min", "sum"),
            normalized_cpu_seconds=("norm_cpu_seconds", "sum"),
        )
        .reset_index()
    )
    daily.insert(0, "month", spec.label)
    totals = {
        "month": spec.label,
        "full_jobs": len(trace),
        "retained_jobs": len(retained),
        "runtime_min": float(retained["runtime_min"].sum()),
        "normalized_cpu_seconds": float(retained["norm_cpu_seconds"].sum()),
        "first_submission": retained["submit_time"].min(),
        "last_submission": retained["submit_time"].max(),
    }
    del trace, retained
    return retained_work, profile, daily, totals


def load_retained_run(
    spec: MonthSpec,
    policy: str,
    path: Path,
    site_info: dict[str, dict[str, float | int | str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = [
        "job_id",
        "submit_time",
        "start_time",
        "finish_time",
        "site",
        "norm_cpu_seconds",
        "queue_delay_min",
        "site_rank_at_start",
        "environmental_time_mode",
        "total_energy_kwh",
        "assigned_ci_gco2_per_kwh",
        "assigned_hi_stress_l_per_kwh",
        "water_impact_stress_l",
    ]
    if not path.exists():
        raise FileNotFoundError(path)

    retained_chunks: list[pd.DataFrame] = []
    full_count = 0
    full_ids: set[int] = set()
    duplicate_ids = 0
    sites: set[str] = set()
    modes: set[str] = set()
    first_submission: pd.Timestamp | None = None
    last_submission: pd.Timestamp | None = None

    for chunk in pd.read_csv(path, usecols=columns, chunksize=150_000):
        full_count += len(chunk)
        ids = chunk["job_id"].astype(int).tolist()
        duplicate_ids += sum(job_id in full_ids for job_id in ids)
        full_ids.update(ids)
        sites.update(chunk["site"].dropna().astype(str).unique())
        modes.update(chunk["environmental_time_mode"].dropna().astype(str).unique())
        chunk["submit_time"] = pd.to_datetime(chunk["submit_time"], errors="raise")
        current_first = chunk["submit_time"].min()
        current_last = chunk["submit_time"].max()
        first_submission = (
            current_first if first_submission is None else min(first_submission, current_first)
        )
        last_submission = (
            current_last if last_submission is None else max(last_submission, current_last)
        )
        mask = (
            (chunk["submit_time"] >= spec.analysis_start)
            & (chunk["submit_time"] < spec.analysis_end)
        )
        retained_chunks.append(chunk.loc[mask].copy())

    if full_count != spec.expected_full_jobs:
        raise ValueError(f"{spec.label} {policy}: unexpected completed-job count")
    if duplicate_ids or len(full_ids) != full_count:
        raise ValueError(f"{spec.label} {policy}: duplicate completed-job IDs")
    if not sites or not sites.issubset(set(SITE_ORDER)):
        raise ValueError(f"{spec.label} {policy}: unexpected site set {sorted(sites)}")
    if modes != {"historical_interval_average"}:
        raise ValueError(f"{spec.label} {policy}: unexpected timing modes {modes}")

    frame = pd.concat(retained_chunks, ignore_index=True)
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="raise")
    frame["finish_time"] = pd.to_datetime(frame["finish_time"], errors="raise")
    frame["month"] = spec.label
    frame["policy"] = policy
    frame["pue"] = frame["site"].map(lambda site: float(site_info[site]["pue"]))
    frame["facility_energy_kwh"] = frame["total_energy_kwh"] * frame["pue"]
    frame["facility_carbon_kg"] = (
        frame["facility_energy_kwh"] * frame["assigned_ci_gco2_per_kwh"] / 1000.0
    )
    frame["recomputed_water_stress_l"] = (
        frame["facility_energy_kwh"] * frame["assigned_hi_stress_l_per_kwh"]
    )
    if not np.allclose(
        frame["water_impact_stress_l"],
        frame["recomputed_water_stress_l"],
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError(f"{spec.label} {policy}: water-accounting mismatch")
    frame["turnaround_min"] = (
        (frame["finish_time"] - frame["submit_time"]).dt.total_seconds() / 60.0
    )
    metadata = {
        "month": spec.label,
        "policy": policy,
        "run_file": str(path.relative_to(ROOT)),
        "full_jobs": full_count,
        "full_unique_job_ids": len(full_ids),
        "full_first_submission": first_submission,
        "full_last_submission": last_submission,
        "retained_jobs": len(frame),
        "retained_unique_job_ids": frame["job_id"].nunique(),
        "retained_first_submission": frame["submit_time"].min(),
        "retained_last_submission": frame["submit_time"].max(),
        "environmental_time_mode": ";".join(sorted(modes)),
        "sites_receiving_jobs": ";".join(sorted(sites)),
        "water_reconciliation_passed": True,
    }
    return frame, metadata


def validate_cohort(
    spec: MonthSpec,
    trace_work: pd.Series,
    random_df: pd.DataFrame,
    green_df: pd.DataFrame,
) -> None:
    for policy, frame in (("Randomized", random_df), ("GreenScore-based", green_df)):
        work = frame.set_index("job_id")["norm_cpu_seconds"].sort_index()
        if not work.index.equals(trace_work.index):
            raise ValueError(f"{spec.label} {policy}: retained IDs differ from the trace")
        if not np.allclose(
            work.to_numpy(), trace_work.to_numpy(), rtol=0, atol=1e-9
        ):
            raise ValueError(f"{spec.label} {policy}: retained CPU demand differs from the trace")
    random_work = random_df.set_index("job_id")["norm_cpu_seconds"].sort_index()
    green_work = green_df.set_index("job_id")["norm_cpu_seconds"].sort_index()
    if not random_work.index.equals(green_work.index):
        raise ValueError(f"{spec.label}: policies do not contain identical job IDs")
    if not np.allclose(
        random_work.to_numpy(), green_work.to_numpy(), rtol=0, atol=1e-9
    ):
        raise ValueError(f"{spec.label}: policies do not contain identical CPU demand")


def summarize_frame(
    spec: MonthSpec,
    policy: str,
    frame: pd.DataFrame,
    site_info: dict[str, dict[str, float | int | str]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cpu = float(frame["norm_cpu_seconds"].sum())
    facility = float(frame["facility_energy_kwh"].sum())
    carbon = float(frame["facility_carbon_kg"].sum())
    water = float(frame["recomputed_water_stress_l"].sum())
    row = {
        "month": spec.label,
        "policy": policy,
        "jobs": len(frame),
        "normalized_cpu_seconds": cpu,
        "it_energy_kwh": float(frame["total_energy_kwh"].sum()),
        "facility_energy_kwh": facility,
        "carbon_kgco2e": carbon,
        "water_stress_l": water,
        "cpu_per_carbon_million": cpu / carbon / 1e6,
        "cpu_per_water_million": cpu / water / 1e6,
        "queue_delay_sum_min": float(frame["queue_delay_min"].sum()),
        "mean_queue_delay_min": float(frame["queue_delay_min"].mean()),
        "median_queue_delay_min": float(frame["queue_delay_min"].median()),
        "p95_queue_delay_min": float(frame["queue_delay_min"].quantile(0.95)),
        "turnaround_sum_min": float(frame["turnaround_min"].sum()),
        "mean_turnaround_min": float(frame["turnaround_min"].mean()),
        "effective_ci_gco2e_per_kwh": carbon * 1000.0 / facility,
        "effective_wi_stress_l_per_kwh": water / facility,
    }
    site_rows: list[dict[str, object]] = []
    for site in SITE_ORDER:
        group = frame[frame["site"] == site]
        site_cpu = float(group["norm_cpu_seconds"].sum())
        site_facility = float(group["facility_energy_kwh"].sum())
        site_carbon = float(group["facility_carbon_kg"].sum())
        site_water = float(group["recomputed_water_stress_l"].sum())
        site_rows.append(
            {
                "month": spec.label,
                "policy": policy,
                "site": site,
                "country": site_info[site]["country"],
                "greenscore_rank": site_info[site]["rank"],
                "configured_greenscore": site_info[site]["score"],
                "pue": site_info[site]["pue"],
                "jobs": len(group),
                "job_share_percent": len(group) / len(frame) * 100.0,
                "normalized_cpu_seconds": site_cpu,
                "cpu_share_percent": site_cpu / cpu * 100.0,
                "it_energy_kwh": float(group["total_energy_kwh"].sum()),
                "facility_energy_kwh": site_facility,
                "carbon_kgco2e": site_carbon,
                "water_stress_l": site_water,
                "effective_ci_gco2e_per_kwh": (
                    site_carbon * 1000.0 / site_facility if site_facility > 0 else np.nan
                ),
                "effective_wi_stress_l_per_kwh": (
                    site_water / site_facility if site_facility > 0 else np.nan
                ),
                "mean_queue_delay_min": (
                    float(group["queue_delay_min"].mean()) if len(group) else np.nan
                ),
            }
        )
    return row, site_rows


def summarize_hourly(
    spec: MonthSpec, policy: str, frame: pd.DataFrame
) -> pd.DataFrame:
    hour_index = (
        (frame["submit_time"] - spec.analysis_start).dt.total_seconds() // 3600
    ).astype(int)
    hourly = (
        frame.assign(hour_index=hour_index)
        .groupby("hour_index")
        .agg(
            jobs=("job_id", "size"),
            normalized_cpu_seconds=("norm_cpu_seconds", "sum"),
            carbon_kgco2e=("facility_carbon_kg", "sum"),
            water_stress_l=("recomputed_water_stress_l", "sum"),
        )
        .reindex(range(120), fill_value=0.0)
    )
    cpu = hourly["normalized_cpu_seconds"]
    carbon = hourly["carbon_kgco2e"]
    water = hourly["water_stress_l"]
    hourly["cpu_per_carbon_million_1h"] = np.divide(
        cpu, carbon, out=np.full(120, np.nan), where=carbon > 0
    ) / 1e6
    hourly["cpu_per_water_million_1h"] = np.divide(
        cpu, water, out=np.full(120, np.nan), where=water > 0
    ) / 1e6
    cpu3 = cpu.rolling(3, center=True, min_periods=3).sum()
    carbon3 = carbon.rolling(3, center=True, min_periods=3).sum()
    water3 = water.rolling(3, center=True, min_periods=3).sum()
    hourly["cpu_per_carbon_million_3h"] = np.divide(
        cpu3, carbon3, out=np.full(120, np.nan), where=carbon3 > 0
    ) / 1e6
    hourly["cpu_per_water_million_3h"] = np.divide(
        cpu3, water3, out=np.full(120, np.nan), where=water3 > 0
    ) / 1e6
    hourly = hourly.reset_index()
    hourly.insert(0, "policy", policy)
    hourly.insert(0, "month", spec.label)
    hourly["elapsed_day_midpoint"] = (hourly["hour_index"] + 0.5) / 24.0
    return hourly


def make_pooled_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        subset = monthly[monthly["policy"] == policy]
        jobs = int(subset["jobs"].sum())
        cpu = float(subset["normalized_cpu_seconds"].sum())
        facility = float(subset["facility_energy_kwh"].sum())
        carbon = float(subset["carbon_kgco2e"].sum())
        water = float(subset["water_stress_l"].sum())
        rows.append(
            {
                "month": "Pooled",
                "policy": policy,
                "jobs": jobs,
                "normalized_cpu_seconds": cpu,
                "it_energy_kwh": float(subset["it_energy_kwh"].sum()),
                "facility_energy_kwh": facility,
                "carbon_kgco2e": carbon,
                "water_stress_l": water,
                "cpu_per_carbon_million": cpu / carbon / 1e6,
                "cpu_per_water_million": cpu / water / 1e6,
                "queue_delay_sum_min": float(subset["queue_delay_sum_min"].sum()),
                "mean_queue_delay_min": float(subset["queue_delay_sum_min"].sum()) / jobs,
                "turnaround_sum_min": float(subset["turnaround_sum_min"].sum()),
                "mean_turnaround_min": float(subset["turnaround_sum_min"].sum()) / jobs,
                "effective_ci_gco2e_per_kwh": carbon * 1000.0 / facility,
                "effective_wi_stress_l_per_kwh": water / facility,
            }
        )
    return pd.DataFrame(rows)


def make_pooled_site_summary(site_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        site_summary.groupby(["policy", "site"], as_index=False)
        .agg(
            jobs=("jobs", "sum"),
            normalized_cpu_seconds=("normalized_cpu_seconds", "sum"),
            facility_energy_kwh=("facility_energy_kwh", "sum"),
            carbon_kgco2e=("carbon_kgco2e", "sum"),
            water_stress_l=("water_stress_l", "sum"),
        )
    )
    totals = grouped.groupby("policy")["normalized_cpu_seconds"].transform("sum")
    grouped["cpu_share_percent"] = grouped["normalized_cpu_seconds"] / totals * 100.0
    grouped.insert(0, "month", "Pooled")
    return grouped


def compute_window_comparisons(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in [spec.label for spec in MONTHS]:
        for metric in ("carbon", "water"):
            column = f"cpu_per_{metric}_million_3h"
            pivot = (
                hourly[hourly["month"] == month]
                .pivot(index="hour_index", columns="policy", values=column)
            )
            valid = pivot[list(POLICIES)].notna().all(axis=1)
            rows.append(
                {
                    "month": month,
                    "metric": metric,
                    "comparable_windows": int(valid.sum()),
                    "greenscore_better_windows": int(
                        (
                            pivot.loc[valid, "GreenScore-based"]
                            > pivot.loc[valid, "Randomized"]
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_workloads(profile: pd.DataFrame, workload_totals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(MONTHS), figsize=(15.2, 5.0), sharex=True)
    blue = "#2166AC"
    totals = workload_totals.set_index("month")
    for col, spec in enumerate(MONTHS):
        subset = profile[profile["month"] == spec.label]
        x = subset["elapsed_days"]
        axes[0, col].plot(
            x, subset["jobs_moving_average_15min"], color=blue, linewidth=1.55
        )
        axes[1, col].plot(
            x,
            subset["runtime_min_moving_average_15min"],
            color=blue,
            linewidth=1.55,
        )
        axes[0, col].set_title(
            f"{spec.label}  ({int(totals.loc[spec.label, 'retained_jobs']):,} jobs)",
            fontweight="bold",
        )
        axes[1, col].set_xlabel("Elapsed day")
        for row in range(2):
            axes[row, col].set_xlim(0, 5)
            axes[row, col].set_xticks(range(6))
            axes[row, col].grid(axis="y", color="#D9D9D9", linewidth=0.65)
    axes[0, 0].set_ylabel("Job arrivals/min")
    axes[1, 0].set_ylabel("Runtime demand\n(min/min)")
    fig.text(0.008, 0.967, "A", fontweight="bold", fontsize=12)
    fig.text(0.008, 0.485, "B", fontweight="bold", fontsize=12)
    fig.tight_layout(h_pad=1.1, w_pad=1.2)
    save_figure(fig, "monthly_workload_profiles")


def plot_environmental_outcomes(monthly: pd.DataFrame) -> None:
    indexed = monthly.set_index(["month", "policy"])
    x = np.arange(len(MONTHS))
    width = 0.35
    metrics = (
        ("facility_energy_kwh", "Facility energy (kWh)", "A"),
        ("carbon_kgco2e", "Carbon footprint (kgCO$_2$e)", "B"),
        ("water_stress_l", "Water-scarcity impact (stress-L)", "C"),
        ("mean_queue_delay_min", "Mean queue delay (min)", "D"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0))
    for axis, (column, ylabel, panel) in zip(axes.flat, metrics):
        random_values = np.array(
            [indexed.loc[(spec.label, "Randomized"), column] for spec in MONTHS]
        )
        green_values = np.array(
            [indexed.loc[(spec.label, "GreenScore-based"), column] for spec in MONTHS]
        )
        axis.bar(
            x - width / 2,
            random_values,
            width,
            color=POLICY_COLORS["Randomized"],
            label="Randomized",
        )
        bars = axis.bar(
            x + width / 2,
            green_values,
            width,
            color=POLICY_COLORS["GreenScore-based"],
            label="GreenScore-based",
        )
        axis.set_xticks(x, [spec.label for spec in MONTHS])
        axis.set_ylabel(ylabel)
        axis.set_title(panel, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65)
        upper = max(random_values.max(), green_values.max()) * 1.24
        axis.set_ylim(0, upper)
        changes = 100.0 * (green_values - random_values) / random_values
        for bar, value, change in zip(bars, green_values, changes):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper * 0.025,
                f"{change:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=POLICY_COLORS["GreenScore-based"],
                fontweight="bold",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.45, w_pad=1.5)
    save_figure(fig, "monthly_environmental_outcomes")


def plot_efficiencies(monthly: pd.DataFrame, pooled: pd.DataFrame) -> None:
    combined = pd.concat([monthly, pooled], ignore_index=True, sort=False)
    indexed = combined.set_index(["month", "policy"])
    categories = [spec.label for spec in MONTHS] + ["Pooled"]
    x = np.arange(len(categories))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    for axis, column, ylabel, panel in (
        (
            axes[0],
            "cpu_per_carbon_million",
            "M normalized CPU-s/kgCO$_2$e",
            "A   Carbon efficiency",
        ),
        (
            axes[1],
            "cpu_per_water_million",
            "M normalized CPU-s/stress-L",
            "B   Water-scarcity efficiency",
        ),
    ):
        random_values = np.array(
            [indexed.loc[(month, "Randomized"), column] for month in categories]
        )
        green_values = np.array(
            [indexed.loc[(month, "GreenScore-based"), column] for month in categories]
        )
        axis.axvspan(2.56, 3.44, color="#EAF4EF", zorder=0)
        axis.bar(
            x - width / 2,
            random_values,
            width,
            color=POLICY_COLORS["Randomized"],
            label="Randomized",
        )
        bars = axis.bar(
            x + width / 2,
            green_values,
            width,
            color=POLICY_COLORS["GreenScore-based"],
            label="GreenScore-based",
        )
        axis.set_xticks(x, categories)
        axis.set_ylabel(ylabel)
        axis.set_title(panel, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65)
        upper = max(random_values.max(), green_values.max()) * 1.27
        axis.set_ylim(0, upper)
        improvements = 100.0 * (green_values - random_values) / random_values
        for bar, value, improvement in zip(bars, green_values, improvements):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper * 0.025,
                f"{improvement:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.4,
                color=POLICY_COLORS["GreenScore-based"],
                fontweight="bold",
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.8)
    save_figure(fig, "monthly_environmental_efficiency")


def plot_time_resolved(hourly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(MONTHS), figsize=(15.2, 5.25), sharex=True)
    for col, spec in enumerate(MONTHS):
        for row, (metric, ylabel) in enumerate(
            (
                ("carbon", "M CPU-s/kgCO$_2$e"),
                ("water", "M CPU-s/stress-L"),
            )
        ):
            axis = axes[row, col]
            for policy in POLICIES:
                subset = hourly[
                    (hourly["month"] == spec.label) & (hourly["policy"] == policy)
                ]
                axis.plot(
                    subset["elapsed_day_midpoint"],
                    subset[f"cpu_per_{metric}_million_1h"],
                    linestyle="none",
                    marker="o",
                    markersize=2.5,
                    alpha=0.20,
                    color=POLICY_COLORS[policy],
                )
                axis.plot(
                    subset["elapsed_day_midpoint"],
                    subset[f"cpu_per_{metric}_million_3h"],
                    linewidth=1.75,
                    color=POLICY_COLORS[policy],
                    label=policy,
                )
            if row == 0:
                axis.set_title(spec.label, fontweight="bold")
            if col == 0:
                axis.set_ylabel(ylabel)
            if row == 1:
                axis.set_xlabel("Elapsed day")
            axis.set_xlim(0, 5)
            axis.set_xticks(range(6))
            axis.grid(color="#D9D9D9", linewidth=0.6)
    fig.text(0.007, 0.965, "A   Carbon efficiency", fontweight="bold", fontsize=11.5)
    fig.text(0.007, 0.485, "B   Water-scarcity efficiency", fontweight="bold", fontsize=11.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.15, w_pad=1.0)
    save_figure(fig, "time_resolved_efficiency_four_months")


def plot_placement(site_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(MONTHS), figsize=(13.2, 3.55), sharey=True)
    for axis, spec in zip(axes, MONTHS):
        bottoms = np.zeros(2)
        for site in SITE_ORDER:
            values = np.array(
                [
                    float(
                        site_summary[
                            (site_summary["month"] == spec.label)
                            & (site_summary["policy"] == policy)
                            & (site_summary["site"] == site)
                        ].iloc[0]["cpu_share_percent"]
                    )
                    for policy in POLICIES
                ]
            )
            axis.bar(
                [0, 1],
                values,
                bottom=bottoms,
                width=0.64,
                color=SITE_COLORS[site],
                label=site,
            )
            bottoms += values
        axis.set_title(spec.label, fontweight="bold")
        axis.set_xticks([0, 1], ["Random", "GreenScore"])
        axis.set_ylim(0, 100)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65)
    axes[0].set_ylabel("Normalized CPU placement share (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.2)
    save_figure(fig, "monthly_site_placement")


def write_latex_values(
    monthly: pd.DataFrame,
    pooled: pd.DataFrame,
    pooled_site: pd.DataFrame,
    workload_totals: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    monthly_indexed = monthly.set_index(["month", "policy"])
    pooled_indexed = pooled.set_index("policy")
    random = pooled_indexed.loc["Randomized"]
    green = pooled_indexed.loc["GreenScore-based"]

    def reduction(random_row: pd.Series, green_row: pd.Series, column: str) -> float:
        return 100.0 * (random_row[column] - green_row[column]) / random_row[column]

    def improvement(random_row: pd.Series, green_row: pd.Series, column: str) -> float:
        return 100.0 * (green_row[column] - random_row[column]) / random_row[column]

    def pooled_share(policy: str, site: str) -> float:
        return float(
            pooled_site[
                (pooled_site["policy"] == policy) & (pooled_site["site"] == site)
            ].iloc[0]["cpu_share_percent"]
        )

    random_top_two = pooled_share("Randomized", "IN2P3-IRES") + pooled_share(
        "Randomized", "SARA-MATRIX"
    )
    green_top_two = pooled_share("GreenScore-based", "IN2P3-IRES") + pooled_share(
        "GreenScore-based", "SARA-MATRIX"
    )
    values: dict[str, str] = {}

    def integer(name: str, value: int | float) -> None:
        values[name] = f"{int(value):,}".replace(",", "{,}")

    def decimal(name: str, value: float, digits: int = 2, signed: bool = False) -> None:
        sign = "+" if signed else ""
        values[name] = f"{float(value):{sign}.{digits}f}"

    integer("totalfulljobs", sum(spec.expected_full_jobs for spec in MONTHS))
    integer("totalcohortjobs", int(green["jobs"]))
    decimal("totalcpubillion", green["normalized_cpu_seconds"] / 1e9, 3)
    decimal("totalruntimemillion", workload_totals["runtime_min"].sum() / 1e6, 3)
    decimal("pooledenergyreduction", reduction(random, green, "facility_energy_kwh"))
    decimal("pooledcarbonreduction", reduction(random, green, "carbon_kgco2e"))
    decimal("pooledwaterreduction", reduction(random, green, "water_stress_l"))
    decimal(
        "pooledcarbonefficiencygain",
        improvement(random, green, "cpu_per_carbon_million"),
    )
    decimal(
        "pooledwaterefficiencygain",
        improvement(random, green, "cpu_per_water_million"),
    )
    decimal(
        "pooledqueuechange",
        100.0
        * (green["mean_queue_delay_min"] - random["mean_queue_delay_min"])
        / random["mean_queue_delay_min"],
        signed=True,
    )
    for prefix, row in (("random", random), ("green", green)):
        decimal(f"pooled{prefix}facility", row["facility_energy_kwh"], 1)
        decimal(f"pooled{prefix}carbon", row["carbon_kgco2e"], 2)
        decimal(f"pooled{prefix}water", row["water_stress_l"], 2)
        decimal(f"pooled{prefix}cpucarbon", row["cpu_per_carbon_million"], 2)
        decimal(f"pooled{prefix}cpuwater", row["cpu_per_water_million"], 2)
        decimal(f"pooled{prefix}wait", row["mean_queue_delay_min"], 2)

    lower_carbon = 0
    lower_water = 0
    for spec in MONTHS:
        random_month = monthly_indexed.loc[(spec.label, "Randomized")]
        green_month = monthly_indexed.loc[(spec.label, "GreenScore-based")]
        prefix = spec.key
        integer(f"{prefix}jobs", int(green_month["jobs"]))
        decimal(
            f"{prefix}energyreduction",
            reduction(random_month, green_month, "facility_energy_kwh"),
        )
        carbon_reduction = reduction(random_month, green_month, "carbon_kgco2e")
        water_reduction = reduction(random_month, green_month, "water_stress_l")
        decimal(f"{prefix}carbonreduction", carbon_reduction)
        decimal(f"{prefix}waterreduction", water_reduction)
        decimal(
            f"{prefix}queuechange",
            100.0
            * (
                green_month["mean_queue_delay_min"]
                - random_month["mean_queue_delay_min"]
            )
            / random_month["mean_queue_delay_min"],
            signed=True,
        )
        lower_carbon += int(carbon_reduction > 0)
        lower_water += int(water_reduction > 0)
    integer("monthslowercarbon", lower_carbon)
    integer("monthslowerwater", lower_water)
    integer("carbonbetterwindows", comparisons.query("metric == 'carbon'")["greenscore_better_windows"].sum())
    integer("carbonwindows", comparisons.query("metric == 'carbon'")["comparable_windows"].sum())
    integer("waterbetterwindows", comparisons.query("metric == 'water'")["greenscore_better_windows"].sum())
    integer("waterwindows", comparisons.query("metric == 'water'")["comparable_windows"].sum())
    decimal("pooledrandomtoptwo", random_top_two)
    decimal("pooledgreentoptwo", green_top_two)
    decimal("pooledrandomtr", pooled_share("Randomized", "TR-03-METU"))
    decimal("pooledgreentr", pooled_share("GreenScore-based", "TR-03-METU"))

    with (DATA / "latex_values.tex").open("w", encoding="utf-8") as handle:
        for name, rendered in values.items():
            handle.write(f"\\newcommand{{\\{name}}}{{{rendered}}}\n")


def write_documentation(
    workload_totals: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    windows = []
    for spec in MONTHS:
        retained_jobs = int(
            workload_totals.loc[workload_totals["month"] == spec.label, "retained_jobs"].iloc[0]
        )
        windows.append(
            {
                "month": spec.label,
                "full_trace_start": spec.full_start,
                "full_trace_end_exclusive": spec.full_end,
                "analysis_start": spec.analysis_start,
                "analysis_end_exclusive": spec.analysis_end,
                "inclusion_timestamp": "submit_time",
                "first_day_treatment": "warm-up boundary; excluded",
                "seventh_day_treatment": "terminal boundary; excluded",
                "impact_accounting": "complete job-level impacts for retained cohort",
                "full_jobs": spec.expected_full_jobs,
                "retained_jobs": retained_jobs,
            }
        )
    pd.DataFrame(windows).to_csv(DATA / "analysis_windows.csv", index=False)
    validation.to_csv(DATA / "run_validation.csv", index=False)

    readme = """# March--June boundary-trimmed DIRAC A0 poster

This Overleaf-ready A0 portrait poster compares paired randomized and
GreenScore-based simulations for four independent seven-day DIRAC traces from
March, April, May, and June 2026. For every trace, the first day is treated as a
warm-up boundary and the seventh as a terminal boundary. The reported cohort
contains only jobs with `submit_time` during days 2--6, represented by the
half-open interval from day 2 at 00:00 through day 7 at 00:00.

Complete job-level energy and impact are retained for every selected job, even
when execution crosses a boundary. Each monthly policy pair contains exactly
the same retained job IDs and normalized CPU demand. Time-resolved ratios are
ratios of sums within submission-hour windows; they are not averages of
per-job ratios.

Facility energy is IT energy times site PUE. Carbon is recomputed as facility
energy times interval-aligned carbon intensity. Water impact is independently
recomputed as facility energy times interval-aligned AWARE-weighted water
intensity and reconciled against the simulator output.

The pooled result is a ratio of sums across the four retained monthly cohorts.
It is not an unweighted average of monthly efficiencies. The randomized policy
uses one fixed seed (42), so the package does not quantify random-seed
uncertainty.

Upload the ZIP to Overleaf and compile `main.tex` with pdfLaTeX. The page is A0
portrait (841 x 1189 mm); print at actual size/100%.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {OUT}")
    FIGURES.mkdir(parents=True)
    DATA.mkdir(parents=True)
    configure_style()
    site_info = read_site_configuration()

    monthly_rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []
    hourly_frames: list[pd.DataFrame] = []
    workload_profiles: list[pd.DataFrame] = []
    workload_days: list[pd.DataFrame] = []
    workload_totals_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for spec in MONTHS:
        print(f"Reading and validating {spec.label} trace...")
        trace_work, profile, daily, workload_totals = read_trace(spec)
        workload_profiles.append(profile)
        workload_days.append(daily)
        workload_totals_rows.append(workload_totals)

        frames: dict[str, pd.DataFrame] = {}
        for policy, path in (
            ("Randomized", spec.random_run),
            ("GreenScore-based", spec.green_run),
        ):
            print(f"Reading {spec.label} {policy} run...")
            frames[policy], metadata = load_retained_run(spec, policy, path, site_info)
            validation_rows.append(metadata)
        validate_cohort(spec, trace_work, frames["Randomized"], frames["GreenScore-based"])

        for policy in POLICIES:
            row, sites = summarize_frame(spec, policy, frames[policy], site_info)
            monthly_rows.append(row)
            site_rows.extend(sites)
            hourly_frames.append(summarize_hourly(spec, policy, frames[policy]))
        del frames, trace_work

    monthly = pd.DataFrame(monthly_rows)
    site_summary = pd.DataFrame(site_rows)
    hourly = pd.concat(hourly_frames, ignore_index=True)
    workload_profile = pd.concat(workload_profiles, ignore_index=True)
    workload_daily = pd.concat(workload_days, ignore_index=True)
    workload_totals = pd.DataFrame(workload_totals_rows)
    validation = pd.DataFrame(validation_rows)
    pooled = make_pooled_summary(monthly)
    pooled_site = make_pooled_site_summary(site_summary)
    comparisons = compute_window_comparisons(hourly)

    for spec in MONTHS:
        trace_cpu = float(
            workload_totals.loc[
                workload_totals["month"] == spec.label, "normalized_cpu_seconds"
            ].iloc[0]
        )
        policy_subset = monthly[monthly["month"] == spec.label]
        if not np.allclose(
            policy_subset["normalized_cpu_seconds"], trace_cpu, rtol=0, atol=1e-6
        ):
            raise ValueError(f"{spec.label}: aggregate CPU does not reconcile to trace")
        retained_jobs = int(
            workload_totals.loc[workload_totals["month"] == spec.label, "retained_jobs"].iloc[0]
        )
        if set(policy_subset["jobs"].astype(int)) != {retained_jobs}:
            raise ValueError(f"{spec.label}: aggregate job count does not reconcile to trace")

    monthly.to_csv(DATA / "aggregate_results_by_month.csv", index=False, float_format="%.9f")
    pooled.to_csv(DATA / "pooled_results.csv", index=False, float_format="%.9f")
    site_summary.to_csv(DATA / "site_results_by_month.csv", index=False, float_format="%.9f")
    pooled_site.to_csv(DATA / "pooled_site_results.csv", index=False, float_format="%.9f")
    hourly.to_csv(DATA / "time_resolved_efficiency.csv", index=False, float_format="%.9f")
    comparisons.to_csv(DATA / "window_comparisons.csv", index=False)
    workload_profile.to_csv(
        DATA / "workload_minute_profiles.csv", index=False, float_format="%.6f"
    )
    workload_daily.to_csv(DATA / "workload_by_day.csv", index=False, float_format="%.6f")
    workload_totals.to_csv(DATA / "workload_totals.csv", index=False, float_format="%.6f")
    shutil.copy2(SIM / "sites.csv", DATA / "sites.csv")

    plot_workloads(workload_profile, workload_totals)
    plot_environmental_outcomes(monthly)
    plot_efficiencies(monthly, pooled)
    plot_time_resolved(hourly)
    plot_placement(site_summary)
    write_latex_values(monthly, pooled, pooled_site, workload_totals, comparisons)
    write_documentation(workload_totals, validation)

    annual_figure = (
        ROOT
        / "overleaf_dirac_three_day_A0_poster_v2_2026-09-02"
        / "figures"
        / "annual_site_characteristics_compact.pdf"
    )
    annual_data = (
        ROOT
        / "DIRAC_simulation_reproduction_guide_data"
        / "turkey_portugal_annual_site_characteristics.csv"
    )
    shutil.copy2(annual_figure, FIGURES / "annual_site_characteristics.pdf")
    shutil.copy2(annual_data, DATA / "annual_site_characteristics.csv")
    shutil.copy2(TEMPLATE, OUT / "main.tex")

    print("\nMonthly aggregate results:")
    print(monthly.to_string(index=False))
    print("\nPooled results:")
    print(pooled.to_string(index=False))
    print("\nCentered three-hour comparisons:")
    print(comparisons.to_string(index=False))
    print(f"\nGenerated {OUT}")


if __name__ == "__main__":
    main()
