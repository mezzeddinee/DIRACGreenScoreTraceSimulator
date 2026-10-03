from __future__ import annotations

import logging
from collections import Counter
from typing import List

try:
    from .models import Job
except ImportError:  # direct script-style execution fallback
    from models import Job

logger = logging.getLogger(__name__)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    idx = int((len(values) - 1) * p)
    return values[idx]


def print_summary(done_jobs: List[Job]) -> None:
    waits: List[float] = []
    turns: List[float] = []
    carbons: List[float] = []
    water_impacts: List[float] = []
    total_norm_cpu_seconds = 0.0
    site_counts: Counter[str] = Counter()
    site_water_impacts: Counter[str] = Counter()

    for j in done_jobs:
        if j.start_time is not None:
            waits.append((j.start_time - j.submit_time).total_seconds() / 60.0)
        if j.finish_time is not None:
            turns.append((j.finish_time - j.submit_time).total_seconds() / 60.0)
        carbons.append(j.carbon_kg)
        water_impacts.append(j.water_impact_stress_l)
        total_norm_cpu_seconds += j.norm_cpu_seconds
        if j.site:
            site_counts[j.site] += 1
            site_water_impacts[j.site] += j.water_impact_stress_l

    waits.sort()
    turns.sort()
    water_impacts.sort()
    total_carbon = sum(carbons)
    avg_carbon = sum(carbons) / len(carbons) if carbons else 0.0
    total_water_impact = sum(water_impacts)
    avg_water_impact = total_water_impact / len(water_impacts) if water_impacts else 0.0
    logger.info(
        "summary jobs=%d avg_carbon=%.8f total_carbon=%.8f total_norm_cpu_seconds=%.2f",
        len(done_jobs),
        avg_carbon,
        total_carbon,
        total_norm_cpu_seconds,
    )

    print(f"Completed jobs: {len(done_jobs)}")
    print(
        "Wait min p50/p90/p95/p99:",
        f"{percentile(waits,0.50):.6f}/{percentile(waits,0.90):.6f}/{percentile(waits,0.95):.6f}/{percentile(waits,0.99):.6f}",
    )
    print(
        "Turnaround min p50/p90/p95/p99:",
        f"{percentile(turns,0.50):.6f}/{percentile(turns,0.90):.6f}/{percentile(turns,0.95):.6f}/{percentile(turns,0.99):.6f}",
    )
    print(f"Total CFP (kgCO2): {total_carbon:.8f}")
    print(f"Total normalized CPU (s): {total_norm_cpu_seconds:.2f}")
    print(f"Average carbon/job (kgCO2): {avg_carbon:.8f}")
    print(f"Total water impact (stress-L): {total_water_impact:.8f}")
    print(f"Average water impact/job (stress-L): {avg_water_impact:.8f}")
    print(
        "Water impact stress-L/job p50/p90/p95/p99:",
        f"{percentile(water_impacts,0.50):.8f}/{percentile(water_impacts,0.90):.8f}/"
        f"{percentile(water_impacts,0.95):.8f}/{percentile(water_impacts,0.99):.8f}",
    )
    print("Notes:")
    print("- Wait = start_time - submit_time (queue delay), in minutes.")
    print("- Turnaround = finish_time - submit_time (wait + execution), in minutes.")
    print("- p50/p90/p95/p99 are percentile cutoffs over completed jobs.")
    print("- Average carbon/job is the mean simulated job emissions in kgCO2.")
    print("- Water impact is AWARE-weighted scarcity impact, not physical water volume.")
    print("Jobs executed per site:")
    for site, count in sorted(site_counts.items()):
        print(f"- {site}: {count}")
    print("Water impact per site (stress-L):")
    for site, impact in sorted(site_water_impacts.items()):
        print(f"- {site}: {impact:.8f}")
