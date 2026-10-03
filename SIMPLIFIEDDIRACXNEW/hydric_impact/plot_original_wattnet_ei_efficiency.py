from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_GREEN = BASE / "timeseries/green_multisite/completed_jobs.csv"
DEFAULT_RANDOM = BASE / "timeseries/random_multisite/completed_jobs.csv"
DEFAULT_DIR = BASE / "timeseries/original_wattnet_greenscore_comparison"
SITE_ZONE = {
    "SARA-MATRIX": "NL",
    "IN2P3-IRES": "FR",
    "FZK-LCG2": "DE",
    "RAL-LCG2": "GB",
}
GREEN = "#009E73"
ORANGE = "#E69F00"
INK = "#26333d"
GRID = "#d8dde1"


def get_json(url: str, headers: dict[str, str] | None = None) -> object:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=90) as response:
        return json.load(response)


def fetch_wattnet_scores(start: datetime, end: datetime) -> list[dict]:
    base = "https://dashboard.wattnet.eu/api"
    token = get_json(f"{base}/core")["token"]
    params = urlencode(
        {
            "metric": "green-score",
            "scope": "operational",
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "aggregate": "false",
            "use_global": "true",
        }
    )
    result = get_json(
        f"{base}/metrics?{params}", headers={"x-dashboard-token": token}
    )
    if not isinstance(result, list):
        raise ValueError("Unexpected response from WattNet metrics endpoint")
    return result


def floor_quarter_hour(value: datetime) -> datetime:
    return value.replace(minute=(value.minute // 15) * 15, second=0, microsecond=0)


def score_lookup(payload: list[dict]) -> dict[tuple[str, datetime], float]:
    lookup: dict[tuple[str, datetime], float] = {}
    wanted = set(SITE_ZONE.values())
    status: dict[str, set[tuple[bool, str]]] = {}
    for zone_record in payload:
        zone = zone_record.get("zone")
        if zone not in wanted:
            continue
        for series in zone_record.get("series", []):
            state = (bool(series.get("valid")), str(series.get("zone_status")))
            status.setdefault(zone, set()).add(state)
            if not state[0]:
                continue
            for timestamp, score in series.get("values", []):
                when = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
                lookup[(zone, when)] = float(score)
    missing_zones = wanted.difference({zone for zone, _ in lookup})
    if missing_zones:
        raise ValueError(f"No valid WattNet EnvironmentalScore for zones: {missing_zones}")
    print("WattNet status:", status)
    return lookup


def csv_time_extent(paths: list[Path]) -> tuple[datetime, datetime]:
    low: datetime | None = None
    high: datetime | None = None
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                start = datetime.fromisoformat(row["start_time"])
                midpoint = start + timedelta(
                    seconds=float(row["assigned_wallclock_seconds"]) / 2.0
                )
                low = midpoint if low is None or midpoint < low else low
                high = midpoint if high is None or midpoint > high else high
    if low is None or high is None:
        raise ValueError("No completed jobs found")
    return floor_quarter_hour(low), floor_quarter_hour(high) + timedelta(minutes=15)


def load_hourly(
    path: Path, scores: dict[tuple[str, datetime], float]
) -> dict[str, np.ndarray]:
    rows: list[tuple[datetime, datetime, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            site = row["site"]
            zone = SITE_ZONE[site]
            start = datetime.fromisoformat(row["start_time"])
            midpoint = start + timedelta(
                seconds=float(row["assigned_wallclock_seconds"]) / 2.0
            )
            quarter = floor_quarter_hour(midpoint)
            quality = scores[(zone, quarter)]
            hi = float(row["assigned_hi_stress_l_per_kwh"])
            if hi <= 0.0:
                raise ValueError(f"Cannot recover facility energy for {row['job_id']}")
            # The archived simulator output satisfies water = IT energy * PUE * HI.
            facility_energy = float(row["water_impact_stress_l"]) / hi
            # WattNet EnvironmentalScore is environmental quality (100 is best), so its
            # complementary fraction is the corresponding environmental burden.
            environmental_burden = facility_energy * (1.0 - quality / 100.0)
            rows.append(
                (
                    datetime.fromisoformat(row["submit_time"]),
                    datetime.fromisoformat(row["finish_time"]),
                    float(row["norm_cpu_seconds"]),
                    environmental_burden,
                )
            )
    origin = min(row[0] for row in rows)
    bins = np.asarray(
        [max(0, int((row[1] - origin).total_seconds() // 3600)) for row in rows],
        dtype=int,
    )
    size = int(bins.max()) + 1
    return {
        "cpu": np.bincount(bins, weights=[row[2] for row in rows], minlength=size),
        "environmental_burden": np.bincount(
            bins, weights=[row[3] for row in rows], minlength=size
        ),
        "jobs": np.bincount(bins, minlength=size),
    }


def pad(values: np.ndarray, size: int) -> np.ndarray:
    return np.pad(values, (0, size - values.size), constant_values=0.0)


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def centered_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    kernel = np.ones(3, dtype=float)
    return ratio(
        np.convolve(numerator, kernel, mode="same"),
        np.convolve(denominator, kernel, mode="same"),
    )


def main(green_path: Path, random_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    start, end = csv_time_extent([green_path, random_path])
    payload = fetch_wattnet_scores(start, end)
    cache = output_dir / "wattnet_green_score_response.json"
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    scores = score_lookup(payload)

    green = load_hourly(green_path, scores)
    random = load_hourly(random_path, scores)
    size = max(len(green["cpu"]), len(random["cpu"]))
    for dataset in (green, random):
        for key in ("cpu", "environmental_burden", "jobs"):
            dataset[key] = pad(dataset[key], size)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    hours = np.arange(size, dtype=float) + 0.5
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.8)
    for label, dataset, color in (
        ("Green-hydric", green, GREEN),
        ("Random", random, ORANGE),
    ):
        hourly = ratio(dataset["cpu"], dataset["environmental_burden"]) / 1_000_000.0
        rolling = centered_ratio(
            dataset["cpu"], dataset["environmental_burden"]
        ) / 1_000_000.0
        ax.plot(hours, hourly, color=color, linewidth=0.8, marker="o",
                markersize=3.2, alpha=0.28)
        ax.plot(hours, rolling, color=color, linewidth=2.25,
                label=f"{label} (3-h window)")
    ax.set_title("Computation delivered per WattNet environmental burden",
                 loc="left", weight="bold")
    ax.set_xlabel("Elapsed simulation time (h)")
    ax.set_ylabel("Million normalized CPU-s / burden-weighted kWh")
    ax.set_xlim(0.0, float(size))
    ax.legend(loc="best", frameon=True, edgecolor="#c8cdd2")
    fig.text(
        0.5, 0.075,
        "Environmental burden = facility energy × (1 − ES/100).",
        ha="center", fontsize=8.7, color="#53606b",
    )
    fig.text(
        0.5, 0.040,
        "Faint points: 1-h bins; bold curves: centered 3-h ratios. Higher is preferable.",
        ha="center", fontsize=8.7, color="#53606b",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.22)
    output = output_dir / "normcpu_per_wattnet_environmental_burden_over_time"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=320 if suffix == "png" else None)
    plt.close(fig)

    green_eff = (
        green["cpu"].sum() / green["environmental_burden"].sum() / 1_000_000.0
    )
    random_eff = (
        random["cpu"].sum() / random["environmental_burden"].sum() / 1_000_000.0
    )
    green_burden = green["environmental_burden"].sum()
    random_burden = random["environmental_burden"].sum()
    summary = (
        "WattNet EnvironmentalScore-derived burden comparison\n"
        f"WattNet interval: {start.isoformat()} to {end.isoformat()} UTC\n"
        "Score sampled at simulated execution midpoint and floored to 15 minutes.\n"
        "Environmental burden = facility energy * (1 - ES / 100).\n"
        f"Green-hydric burden: {green_burden:.9f} burden-weighted kWh\n"
        f"Random burden: {random_burden:.9f} burden-weighted kWh\n"
        f"Green-hydric efficiency: {green_eff:.6f} M norm CPU-s/burden-weighted kWh\n"
        f"Random efficiency: {random_eff:.6f} M norm CPU-s/burden-weighted kWh\n"
        f"Efficiency change: {(green_eff / random_eff - 1.0) * 100.0:.4f}%\n"
        f"Burden reduction: {(1.0 - green_burden / random_burden) * 100.0:.4f}%\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    for suffix in ("png", "pdf", "svg"):
        print(output.with_suffix(f".{suffix}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--green", type=Path, default=DEFAULT_GREEN)
    parser.add_argument("--random", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    main(args.green, args.random, args.output_dir)
