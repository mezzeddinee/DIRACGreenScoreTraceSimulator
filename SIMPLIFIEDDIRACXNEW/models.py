from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Job:
    job_id: str
    submit_time: datetime
    norm_cpu_seconds: float = 0.0
    cores_used: int = 1
    wallclock: float = 0.0
    cpu_norm_factor: float = 1.0

    status: str = "pending"  # pending/waiting/running/done
    start_time: Optional[datetime] = None
    finish_time: Optional[datetime] = None
    site: Optional[str] = None
    remaining_min: int = 0
    carbon_kg: float = 0.0
    total_energy_kwh: float = 0.0
    assigned_ci_gco2_per_kwh: float = 0.0
    assigned_hi_stress_l_per_kwh: float = 0.0
    water_impact_stress_l: float = 0.0
    assigned_cpu_seconds: float = 0.0
    assigned_wallclock_seconds: float = 0.0
    assigned_runtime_min: int = 0
    site_rank_at_start: int = 0
    spillover_delay_threshold_min: int = 0
    environmental_time_mode: str = ""
    environmental_interval_start: Optional[datetime] = None
    environmental_interval_end: Optional[datetime] = None

    def activate(self) -> None:
        self.status = "waiting"
        self.remaining_min = 0


@dataclass
class Site:
    name: str
    max_running_jobs: int
    green: float
    latitude: Optional[float]
    longitude: Optional[float]
    avg_tdp_w: float
    avg_total_cores: int
    perf_hs06: float
    pue: Optional[float] = None
    greenhydric: float = 0.0
    running_jobs: list[Job] = field(default_factory=list)

    def available_slots(self) -> int:
        return self.max_running_jobs - len(self.running_jobs)

    def running(self) -> list[Job]:
        return list(self.running_jobs)
