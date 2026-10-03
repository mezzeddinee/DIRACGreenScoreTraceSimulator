#!/usr/bin/env python3
"""Build an Overleaf-ready A0 poster comparing two controlled site sets."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "dirac_greenscore_simulator"
RUN_ROOT = SIM / "environmental_analysis" / "timeseries"
OUT = ROOT / "overleaf_dirac_controlled_site_set_comparison_A0_2026-09-05"
FIG = OUT / "figures"
DATA = OUT / "data"
ZIP_PATH = ROOT / f"{OUT.name}.zip"

MONTHS = {
    "03": ("March", "2026-03-02", "2026-03-07"),
    "04": ("April", "2026-04-02", "2026-04-07"),
    "05": ("May", "2026-05-02", "2026-05-07"),
    "06": ("June", "2026-06-02", "2026-06-07"),
}

CONFIGS = {
    "PT-TR": {
        "label": "Portugal--Türkiye configuration",
        "short": "PT--TR",
        "color": "#D97706",
        "sites_file": SIM / "sites.csv",
        "sites": ["IN2P3-IRES", "SARA-MATRIX", "NCG-INGRID-PT", "TR-03-METU"],
        "runs": {
            "03": ("trace_march_7d_random_rerun", "trace_march_7d_greenscore_rerun"),
            "04": ("trace_april_7d_random_rerun", "trace_april_7d_greenscore_rerun"),
            "05": ("trace_may_7d_random_rerun", "trace_may_7d_greenscore_rerun"),
            "06": ("trace_7d_random_rerun", "trace_7d_greenscore_rerun"),
        },
    },
    "DE-UK": {
        "label": "Germany--United Kingdom configuration",
        "short": "DE--UK",
        "color": "#2166AC",
        "sites_file": SIM / "sites_original_four_with_pue.csv",
        "sites": ["IN2P3-IRES", "SARA-MATRIX", "FZK-LCG2", "RAL-LCG2"],
        "runs": {
            month: (
                f"original_four_sites_2026_{month}_random_seed42",
                f"original_four_sites_2026_{month}_greenscore",
            )
            for month in MONTHS
        },
    },
}

COUNTRIES = {
    "IN2P3-IRES": "France",
    "SARA-MATRIX": "Netherlands",
    "NCG-INGRID-PT": "Portugal",
    "TR-03-METU": "Türkiye",
    "FZK-LCG2": "Germany",
    "RAL-LCG2": "United Kingdom",
}
POLICIES = ("Randomized", "GreenScore-based")
USECOLS = [
    "job_id",
    "submit_time",
    "site",
    "norm_cpu_seconds",
    "queue_delay_min",
    "total_energy_kwh",
    "assigned_ci_gco2_per_kwh",
    "assigned_hi_stress_l_per_kwh",
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def read_sites(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["site"]: {
            "pue": float(row["pue"]),
            "gs": float(row["greenhydric"]),
            "tdp": float(row["avg_tdp_w"]),
            "cores": float(row["avg_total_cores"]),
            "cpu_norm": float(row["perf_hs06"]),
        }
        for row in rows
    }


def aggregate_run(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pue: dict[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    total = {
        "jobs": 0.0,
        "cpu": 0.0,
        "it_energy": 0.0,
        "facility_energy": 0.0,
        "carbon": 0.0,
        "water": 0.0,
        "queue_sum": 0.0,
    }
    sites: dict[str, dict[str, float]] = {}

    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=200_000):
        chunk["submit_time"] = pd.to_datetime(chunk["submit_time"], errors="raise")
        chunk = chunk[(chunk["submit_time"] >= start) & (chunk["submit_time"] < end)].copy()
        if chunk.empty:
            continue
        chunk["pue"] = chunk["site"].map(pue)
        if chunk["pue"].isna().any():
            raise ValueError(f"Missing PUE in {path}: {chunk.loc[chunk['pue'].isna(), 'site'].unique()}")
        chunk["facility_energy"] = chunk["total_energy_kwh"] * chunk["pue"]
        chunk["facility_carbon"] = (
            chunk["facility_energy"] * chunk["assigned_ci_gco2_per_kwh"] / 1000.0
        )
        chunk["facility_water"] = (
            chunk["facility_energy"] * chunk["assigned_hi_stress_l_per_kwh"]
        )

        total["jobs"] += len(chunk)
        total["cpu"] += chunk["norm_cpu_seconds"].sum()
        total["it_energy"] += chunk["total_energy_kwh"].sum()
        total["facility_energy"] += chunk["facility_energy"].sum()
        total["carbon"] += chunk["facility_carbon"].sum()
        total["water"] += chunk["facility_water"].sum()
        total["queue_sum"] += chunk["queue_delay_min"].sum()

        grouped = chunk.groupby("site", sort=False).agg(
            jobs=("job_id", "size"),
            cpu=("norm_cpu_seconds", "sum"),
            it_energy=("total_energy_kwh", "sum"),
            facility_energy=("facility_energy", "sum"),
            carbon=("facility_carbon", "sum"),
            water=("facility_water", "sum"),
            ci_energy_numerator=(
                "assigned_ci_gco2_per_kwh",
                lambda values: float(
                    np.dot(values.to_numpy(), chunk.loc[values.index, "total_energy_kwh"].to_numpy())
                ),
            ),
            wi_facility_numerator=("facility_water", "sum"),
        )
        for site, row in grouped.iterrows():
            target = sites.setdefault(
                site,
                {
                    "jobs": 0.0,
                    "cpu": 0.0,
                    "it_energy": 0.0,
                    "facility_energy": 0.0,
                    "carbon": 0.0,
                    "water": 0.0,
                    "ci_energy_numerator": 0.0,
                    "wi_facility_numerator": 0.0,
                },
            )
            for key in target:
                target[key] += float(row[key])

    if not total["jobs"]:
        raise ValueError(f"No retained jobs in {path}")
    total["queue_mean"] = total["queue_sum"] / total["jobs"]
    total["cpu_per_carbon_m"] = total["cpu"] / total["carbon"] / 1e6
    total["cpu_per_water_m"] = total["cpu"] / total["water"] / 1e6
    return total, sites


def annual_characteristics(site_data: dict[str, dict[str, float]]) -> pd.DataFrame:
    source = ROOT / "alvaro_monthly_ei_es_gs_2025-06_2026-07" / "monthly_environmental_score_components.csv"
    frame = pd.read_csv(source)
    months = [f"2025-{month:02d}" for month in range(6, 13)] + [f"2026-{month:02d}" for month in range(1, 6)]
    frame = frame[frame["month"].isin(months)]
    rows = []
    for site in sorted(site_data):
        selected = frame[frame["site"] == site]
        if len(selected) != 12:
            raise ValueError(f"Expected 12 environmental rows for {site}, found {len(selected)}")
        weights = selected["paired_15min_samples"].astype(float)
        rows.append(
            {
                "site": site,
                "country": COUNTRIES[site],
                "mean_ci_gco2e_per_kwh": np.average(selected["mean_cf_gco2e_per_kwh"], weights=weights),
                "mean_wi_stress_l_per_kwh": np.average(selected["mean_hi_stress_l_per_kwh"], weights=weights),
                "mean_cee": selected["monthly_cee"].mean(),
                **site_data[site],
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_site_characteristics(characteristics: pd.DataFrame) -> None:
    order = ["IN2P3-IRES", "SARA-MATRIX", "FZK-LCG2", "RAL-LCG2", "NCG-INGRID-PT", "TR-03-METU"]
    frame = characteristics.set_index("site").loc[order].reset_index()
    colors = ["#4C78A8", "#59A14F", "#76B7B2", "#2F5597", "#F28E2B", "#E15759"]
    labels = [f"{row.site}\n({row.country})" for row in frame.itertuples()]
    panels = [
        ("mean_ci_gco2e_per_kwh", "A  Carbon intensity", "gCO$_2$e kWh$^{-1}$", 1),
        ("mean_wi_stress_l_per_kwh", "B  Water-scarcity intensity", "stress-L kWh$^{-1}$", 3),
        ("mean_cee", "C  CPU energy efficiency", "normalized performance W$^{-1}$", 2),
        ("gs", "D  Static GreenScore", "GreenScore", 1),
    ]
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 4.8), sharey=True)
    for index, (ax, (column, title, xlabel, decimals)) in enumerate(zip(axes, panels)):
        values = frame[column].to_numpy(float)
        bars = ax.barh(y, values, color=colors, height=0.62, edgecolor="white")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xlim(0, values.max() * 1.28)
        for bar, value in zip(bars, values):
            ax.text(value + values.max() * 0.02, bar.get_y() + bar.get_height()/2, f"{value:.{decimals}f}", va="center", fontsize=8.5)
        if index == 0:
            ax.set_yticks(y, labels)
        else:
            ax.tick_params(labelleft=False)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.18, top=0.90, wspace=0.35)
    save_figure(fig, "site_characteristics")


def plot_monthly_efficiency(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)
    x = np.arange(len(MONTHS))
    width = 0.19
    offsets = [-1.5, -0.5, 0.5, 1.5]
    entries = [
        ("PT-TR", "Randomized", "PT--TR Random", "#F4B183", ""),
        ("PT-TR", "GreenScore-based", "PT--TR GreenScore", "#D97706", "//"),
        ("DE-UK", "Randomized", "DE--UK Random", "#9DC3E6", ""),
        ("DE-UK", "GreenScore-based", "DE--UK GreenScore", "#2166AC", "//"),
    ]
    for ax, metric, title, ylabel in [
        (axes[0], "cpu_per_carbon_m", "A  Carbon efficiency", "M normalized CPU-s kgCO$_2$e$^{-1}$"),
        (axes[1], "cpu_per_water_m", "B  Water-scarcity efficiency", "M normalized CPU-s stress-L$^{-1}$"),
    ]:
        for offset, (config, policy, label, color, hatch) in zip(offsets, entries):
            values = [float(summary[(summary.config == config) & (summary.month == month) & (summary.policy == policy)][metric].iloc[0]) for month in MONTHS]
            ax.bar(x + offset * width, values, width, label=label, color=color, hatch=hatch, edgecolor="white")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(ncol=2, frameon=False, loc="upper right")
    axes[1].set_xticks(x, [MONTHS[m][0] for m in MONTHS])
    fig.tight_layout()
    save_figure(fig, "monthly_environmental_efficiency")


def plot_relative_changes(comparisons: pd.DataFrame) -> None:
    pooled = comparisons[comparisons.month == "Pooled"].set_index("config")
    panels = [
        (["carbon_change_pct", "water_change_pct"], ["Carbon", "Water scarcity"], "A  Physical footprints", "Negative is preferable"),
        (["carbon_eff_gain_pct", "water_eff_gain_pct"], ["CPU/carbon", "CPU/water"], "B  Computational efficiency", "Positive is preferable"),
    ]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for ax, (columns, labels, title, direction) in zip(axes, panels):
        for config, offset in [("PT-TR", -0.18), ("DE-UK", 0.18)]:
            values = [float(pooled.loc[config, column]) for column in columns]
            bars = ax.bar(x + offset, values, 0.34, color=CONFIGS[config]["color"], label=CONFIGS[config]["short"])
            for bar, value in zip(bars, values):
                vertical = 5 if value >= 0 else -5
                ax.annotate(
                    f"{value:+.1f}%",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, vertical),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=9,
                    fontweight="bold",
                )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylabel("GreenScore vs random (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.96, direction, transform=ax.transAxes, va="top", fontsize=9, color="#555555")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "relative_policy_effects")


def plot_allocation(site_summary: pd.DataFrame) -> None:
    site_order = ["IN2P3-IRES", "SARA-MATRIX", "FZK-LCG2", "RAL-LCG2", "NCG-INGRID-PT", "TR-03-METU"]
    columns = [("PT-TR", "Randomized"), ("PT-TR", "GreenScore-based"), ("DE-UK", "Randomized"), ("DE-UK", "GreenScore-based")]
    values = np.full((len(site_order), len(columns)), np.nan)
    for j, (config, policy) in enumerate(columns):
        selected = site_summary[(site_summary.config == config) & (site_summary.policy == policy)]
        total = selected.cpu.sum()
        shares = selected.groupby("site").cpu.sum() / total * 100
        for i, site in enumerate(site_order):
            if site in CONFIGS[config]["sites"]:
                values[i, j] = shares.get(site, 0.0)

    fig, ax = plt.subplots(figsize=(9.7, 5.5))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap="YlGnBu", vmin=0, vmax=np.nanmax(values), aspect="auto")
    ax.set_xticks(range(len(columns)), ["PT--TR\nRandom", "PT--TR\nGreenScore", "DE--UK\nRandom", "DE--UK\nGreenScore"])
    ax.set_yticks(range(len(site_order)), [f"{s} ({COUNTRIES[s]})" for s in site_order])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                color = "white" if values[i, j] > np.nanmax(values) * 0.55 else "#17242E"
                ax.text(j, i, f"{values[i,j]:.1f}%", ha="center", va="center", color=color, fontweight="bold")
            else:
                ax.text(j, i, "--", ha="center", va="center", color="#888888")
    ax.set_title("Pooled normalized CPU placement", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Share of normalized CPU work (%)", fraction=0.04, pad=0.03)
    fig.tight_layout()
    save_figure(fig, "pooled_site_allocation")


def pooled_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        for policy in POLICIES:
            selected = summary[(summary.config == config) & (summary.policy == policy)]
            cpu = selected.cpu.sum()
            jobs = selected.jobs.sum()
            rows.append(
                {
                    "config": config,
                    "policy": policy,
                    "jobs": jobs,
                    "cpu": cpu,
                    "facility_energy": selected.facility_energy.sum(),
                    "carbon": selected.carbon.sum(),
                    "water": selected.water.sum(),
                    "queue_mean": np.average(selected.queue_mean, weights=selected.jobs),
                    "cpu_per_carbon_m": cpu / selected.carbon.sum() / 1e6,
                    "cpu_per_water_m": cpu / selected.water.sum() / 1e6,
                }
            )
    return pd.DataFrame(rows)


def comparison_rows(summary: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        for month in list(MONTHS) + ["Pooled"]:
            frame = pooled if month == "Pooled" else summary[summary.month == month]
            random = frame[(frame.config == config) & (frame.policy == "Randomized")].iloc[0]
            green = frame[(frame.config == config) & (frame.policy == "GreenScore-based")].iloc[0]
            rows.append(
                {
                    "config": config,
                    "month": month,
                    "energy_change_pct": (green.facility_energy / random.facility_energy - 1) * 100,
                    "carbon_change_pct": (green.carbon / random.carbon - 1) * 100,
                    "water_change_pct": (green.water / random.water - 1) * 100,
                    "carbon_eff_gain_pct": (green.cpu_per_carbon_m / random.cpu_per_carbon_m - 1) * 100,
                    "water_eff_gain_pct": (green.cpu_per_water_m / random.cpu_per_water_m - 1) * 100,
                    "queue_change_pct": (green.queue_mean / random.queue_mean - 1) * 100 if random.queue_mean else 0,
                }
            )
    return pd.DataFrame(rows)


def latex_command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def write_latex_values(pooled: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    lines = []
    for config, prefix in [("PT-TR", "pttr"), ("DE-UK", "deuk")]:
        comp = comparisons[(comparisons.config == config) & (comparisons.month == "Pooled")].iloc[0]
        lines.extend(
            [
                latex_command(prefix + "carbonchange", f"{comp.carbon_change_pct:+.1f}"),
                latex_command(prefix + "waterchange", f"{comp.water_change_pct:+.1f}"),
                latex_command(prefix + "carbongain", f"{comp.carbon_eff_gain_pct:+.1f}"),
                latex_command(prefix + "watergain", f"{comp.water_eff_gain_pct:+.1f}"),
                latex_command(prefix + "queuechange", f"{comp.queue_change_pct:+.2f}"),
            ]
        )
    jobs = int(pooled.iloc[0].jobs)
    cpu = pooled.iloc[0].cpu / 1e9
    lines.append(latex_command("retainedjobs", f"{jobs:,}"))
    lines.append(latex_command("retainedcpu", f"{cpu:.2f}"))
    (DATA / "latex_values.tex").write_text("".join(lines), encoding="utf-8")


def write_main_tex(pooled: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    table_lines = []
    for row in pooled.itertuples():
        table_lines.append(
            f"{CONFIGS[row.config]['short']} & {row.policy.replace('GreenScore-based', 'GreenScore')} & "
            f"{row.facility_energy:.1f} & {row.carbon:.1f} & {row.water:.1f} & "
            f"{row.cpu_per_carbon_m:.1f} & {row.cpu_per_water_m:.1f} \\\\\n"
        )
    table = "".join(table_lines)
    tex = r"""\documentclass[final]{beamer}
\usepackage[orientation=portrait,size=a0,scale=0.94]{beamerposter}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{ragged2e}
\usefonttheme{professionalfonts}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{blocks}[rounded][shadow=false]
\setbeamersize{text margin left=2cm,text margin right=2cm}
\definecolor{DeepBlue}{HTML}{123A5A}
\definecolor{OceanBlue}{HTML}{2166AC}
\definecolor{Orange}{HTML}{D97706}
\definecolor{Green}{HTML}{228B5A}
\definecolor{WarmGray}{HTML}{F3F2EF}
\definecolor{Ink}{HTML}{17242E}
\setbeamercolor{normal text}{fg=Ink,bg=white}
\setbeamercolor{block title}{fg=white,bg=DeepBlue}
\setbeamercolor{block body}{fg=Ink,bg=WarmGray}
\setbeamerfont{block title}{series=\bfseries,size=\Large}
\graphicspath{{figures/}}
\input{data/latex_values.tex}
\newcommand{\metricbox}[3]{\begin{minipage}[c][4.4cm][c]{0.235\textwidth}\centering{\fontsize{40}{44}\selectfont\bfseries\color{#3}#1\par}\vspace{0.15cm}{\large #2}\end{minipage}}
\begin{document}
\begin{frame}[t]
\begin{beamercolorbox}[wd=\textwidth,sep=0.82cm,center]{block title}
{\fontsize{49}{54}\selectfont\bfseries Controlled DIRAC Site-Set Comparison}\par
\vspace{0.15cm}{\fontsize{26}{31}\selectfont GreenScore-based versus randomized pilot placement across four monthly traces}\par
\vspace{0.30cm}{\Large Mazen Ezzeddine \quad and \quad Andrei Tsaregorodtsev}\par
{\large Aix Marseille Univ, CNRS/IN2P3, CPPM, Marseille, France}
\end{beamercolorbox}
\vspace{0.35cm}
\begin{beamercolorbox}[wd=\textwidth,sep=0.36cm,rounded=true]{block body}\centering
\metricbox{\retainedjobs}{jobs in each paired configuration}{OceanBlue}\hfill
\metricbox{\pttrcarbongain\%}{PT--TR carbon-efficiency change}{Orange}\hfill
\metricbox{\deukcarbongain\%}{DE--UK carbon-efficiency change}{OceanBlue}\hfill
\metricbox{\deukwatergain\%}{DE--UK water-efficiency change}{Orange}
\end{beamercolorbox}
\vspace{0.35cm}

\begin{block}{Controlled design and compared site sets}
\justifying
The same March, April, May, and June 2026 seven-day traces are replayed under randomized ordering (seed 42) and fixed GreenScore ordering. For every run, jobs submitted on day 1 provide warm-up context and jobs submitted on day 7 form the terminal boundary; only the identical day 2--6 cohort is evaluated. The workload, 750-slot site capacities, environmental-time alignment, PUE treatment, and accounting equations are held fixed. The intervention is the site set: both configurations contain IN2P3-IRES (France) and SARA-MATRIX (Netherlands); the PT--TR configuration adds NCG-INGRID-PT (Portugal) and TR-03-METU (Türkiye), whereas the DE--UK configuration adds FZK-LCG2 (Germany) and RAL-LCG2 (United Kingdom).
\vspace{0.18cm}\centering
\includegraphics[width=0.91\linewidth]{site_characteristics.pdf}

{\small Contextual means cover June 2025--May 2026. Lower CI and AWARE-weighted WI are preferable; higher CEE and GreenScore are preferable.}
\end{block}
\vspace{0.35cm}

\begin{columns}[T,totalwidth=\textwidth]
\begin{column}{0.493\textwidth}
\begin{block}{Monthly environmental efficiency}
\centering\includegraphics[width=0.96\linewidth]{monthly_environmental_efficiency.pdf}
\justifying
Each ratio is computed from aggregate normalized CPU work and aggregate physical impact within the retained cohort. The paired workload is identical within each month and configuration; higher values are preferable.
\end{block}
\end{column}
\begin{column}{0.493\textwidth}
\begin{block}{Policy effect within each site set}
\centering\includegraphics[width=0.96\linewidth]{relative_policy_effects.pdf}
\justifying
Changes compare GreenScore with randomized ordering inside the same site configuration. Negative footprint changes and positive efficiency changes indicate improvement. Pooled quantities are ratios of pooled sums, not averages of monthly percentages.
\end{block}
\end{column}
\end{columns}
\vspace{0.35cm}

\begin{columns}[T,totalwidth=\textwidth]
\begin{column}{0.55\textwidth}
\begin{block}{Pooled four-month results}
\centering
\resizebox{0.98\linewidth}{!}{\begin{tabular}{llrrrrr}
\toprule
Site set & Policy & Facility energy & Carbon & Water & CPU/CF & CPU/Water\\
& & (kWh) & (kgCO$_2$e) & (stress-L) & (M CPU-s/kg) & (M CPU-s/stress-L)\\
\midrule
""" + table + r"""\bottomrule
\end{tabular}}
\vspace{0.25cm}
\justifying
All four policy/configuration combinations process the same \retainedjobs{} jobs and \retainedcpu{} billion normalized CPU-seconds. Facility energy equals attributed IT energy multiplied by site PUE; carbon and water use the corresponding interval-aligned WattNet intensities.
\end{block}
\end{column}
\begin{column}{0.443\textwidth}
\begin{block}{Pooled placement by site}
\centering\includegraphics[width=0.96\linewidth]{pooled_site_allocation.pdf}
\end{block}
\end{column}
\end{columns}
\vspace{0.35cm}

\begin{block}{Interpretation}
\justifying
\begin{itemize}
\item In the PT--TR configuration, GreenScore changes pooled carbon and water footprints by \pttrcarbonchange\% and \pttrwaterchange\%, respectively; carbon and water efficiencies change by \pttrcarbongain\% and \pttrwatergain\%.
\item In the DE--UK configuration, the corresponding footprint changes are \deukcarbonchange\% for carbon and \deukwaterchange\% for water, while efficiency changes are \deukcarbongain\% and \deukwatergain\%.
\item The contrast quantifies site-set sensitivity: replacing the Portuguese and Turkish resources with German and UK resources changes both the environmental opportunity and where the fixed GreenScore ranking directs work.
\item Queue-delay changes remain small (\pttrqueuechange\% for PT--TR and \deukqueuechange\% for DE--UK), so the observed environmental differences are not driven by materially different queued demand in this controlled model.
\item Results remain conditional on four monthly workloads, static GreenScores, equal configured capacities, historical interval reconstruction, and one randomized seed.
\end{itemize}
\end{block}
\vfill
\begin{beamercolorbox}[wd=\textwidth,sep=0.28cm,center]{block title}
{\large Controlled comparison: identical traces and accounting; only the eligible site set and policy-specific ordering differ.}
\end{beamercolorbox}
\end{frame}
\end{document}
"""
    (OUT / "main.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    configure_style()

    all_site_data: dict[str, dict[str, float]] = {}
    summary_rows = []
    site_rows = []
    retained_counts: dict[str, int] = {}

    for config, info in CONFIGS.items():
        site_data = read_sites(info["sites_file"])
        if set(site_data) != set(info["sites"]):
            raise ValueError(f"Unexpected site set in {info['sites_file']}")
        all_site_data.update(site_data)
        pue = {site: values["pue"] for site, values in site_data.items()}
        for month, (_, start_text, end_text) in MONTHS.items():
            start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
            random_name, green_name = info["runs"][month]
            month_counts = []
            for policy, run_name in zip(POLICIES, (random_name, green_name)):
                totals, sites = aggregate_run(RUN_ROOT / run_name / "completed_jobs.csv", start, end, pue)
                month_counts.append(int(totals["jobs"]))
                summary_rows.append({"config": config, "month": month, "policy": policy, **totals})
                for site, values in sites.items():
                    site_rows.append({"config": config, "month": month, "policy": policy, "site": site, **values})
            if month_counts[0] != month_counts[1]:
                raise ValueError(f"Policy job-count mismatch for {config} {month}")
            prior = retained_counts.setdefault(month, month_counts[0])
            if prior != month_counts[0]:
                raise ValueError(f"Configuration job-count mismatch for {month}")

    summary = pd.DataFrame(summary_rows)
    site_summary = pd.DataFrame(site_rows)
    pooled = pooled_rows(summary)
    comparisons = comparison_rows(summary, pooled)
    characteristics = annual_characteristics(all_site_data)

    summary.to_csv(DATA / "monthly_aggregate_results.csv", index=False)
    site_summary.to_csv(DATA / "monthly_site_results.csv", index=False)
    pooled.to_csv(DATA / "pooled_aggregate_results.csv", index=False)
    comparisons.to_csv(DATA / "policy_relative_changes.csv", index=False)
    characteristics.to_csv(DATA / "annual_site_characteristics.csv", index=False)

    plot_site_characteristics(characteristics)
    plot_monthly_efficiency(summary)
    plot_relative_changes(comparisons)
    plot_allocation(site_summary)
    write_latex_values(pooled, comparisons)
    write_main_tex(pooled, comparisons)

    readme = """# Controlled DIRAC site-set comparison

Upload the ZIP directly to Overleaf and compile `main.tex` with pdfLaTeX.

The poster compares randomized and GreenScore-based placement for two four-site
configurations across identical March--June 2026 traces. Only jobs submitted on
days 2--6 are reported; days 1 and 7 are retained in simulation as boundaries.
The `data/` directory contains all processed values used by the figures and
table. Carbon and water are recomputed consistently from attributed IT energy,
site PUE, and the saved interval-aligned intensities.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUT))
    desktop = Path.home() / "Desktop" / ZIP_PATH.name
    shutil.copy2(ZIP_PATH, desktop)

    print(f"Package: {OUT}")
    print(f"ZIP: {ZIP_PATH}")
    print(f"Desktop ZIP: {desktop}")
    print("\nPooled results:")
    print(pooled.to_string(index=False))
    print("\nPooled policy changes:")
    print(comparisons[comparisons.month == "Pooled"].to_string(index=False))


if __name__ == "__main__":
    main()
