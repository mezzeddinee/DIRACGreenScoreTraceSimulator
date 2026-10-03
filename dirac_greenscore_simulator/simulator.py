from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    from .models import Job, Site
    from .policy import ReplayCarbonPolicy
except ImportError:  # direct script-style execution fallback
    from models import Job, Site
    from policy import ReplayCarbonPolicy

logger = logging.getLogger(__name__)


class ReplaySimulator:
    def __init__(
        self,
        sites: Dict[str, Site],
        jobs: List[Job],
        tick_minutes: int = 1,
        policy: Optional[ReplayCarbonPolicy] = None,
        ci_provider: Any = None,
        hydric_impact_provider: Any = None,
        ci_timestamp_mode: Optional[str] = None,
        environmental_time_mode: Optional[str] = None,
    ):
        self.sites = sites
        self.pending_jobs = sorted(jobs, key=lambda j: (j.submit_time, j.job_id))
        self.pending_index = 0
        self.waiting_queue: Deque[Job] = deque()
        self.total_jobs = len(self.pending_jobs)
        self.done_count = sum(1 for j in self.pending_jobs if j.status == "done")
        self.policy = policy or ReplayCarbonPolicy()
        self.ci_provider = ci_provider
        if self.ci_provider is None:
            raise ValueError("ci_provider is required")
        self.hydric_impact_provider = hydric_impact_provider
        selected_time_mode = environmental_time_mode or ci_timestamp_mode
        self.environmental_time_mode = (
            selected_time_mode or "historical_interval_average"
        ).strip().lower()
        self.ci_timestamp_mode = self.environmental_time_mode
        if self.environmental_time_mode not in {
            "historical_interval_average",
            "submission",
            "execution_start",
            "execution_midpoint",
        }:
            raise ValueError(
                "environmental_time_mode must be historical_interval_average, "
                "submission, execution_start, or execution_midpoint"
            )
        self.tick = timedelta(minutes=tick_minutes)
        self.current_time = min(j.submit_time for j in jobs)
        self.done_jobs: List[Job] = []
        self.idle_consumption_factor = 0.4
        self.waiting_history: List[Tuple[datetime, int]] = []
        self.submissions_history: List[Tuple[datetime, Dict[str, int]]] = []
        self.running_history: List[Tuple[datetime, Dict[str, int]]] = []
        logger.info(
            "sim init sites=%d jobs=%d tick_min=%d environmental_time_mode=%s",
            len(self.sites),
            len(self.pending_jobs),
            tick_minutes,
            self.environmental_time_mode,
        )

    def waiting_jobs(self) -> List[Job]:
        return list(self.waiting_queue)

    def active_jobs(self) -> int:
        return sum(len(s.running_jobs) for s in self.sites.values())

    def running_jobs_by_site(self) -> Dict[str, int]:
        return {name: len(site.running_jobs) for name, site in sorted(self.sites.items())}

    def release_jobs(self) -> None:
        # Check all jobs that are not released yet.
        # If their submit time is now, put them in the waiting queue.
        released = 0
        while self.pending_index < self.total_jobs:
            j = self.pending_jobs[self.pending_index]
            if j.submit_time > self.current_time:
                break
            if j.status == "pending":
                j.activate()
                self.waiting_queue.append(j)
                released += 1
            self.pending_index += 1
        if released:
            logger.info("release t=%s jobs=%d", self.current_time.isoformat(), released)

    def _historical_interval(self, job: Job) -> Tuple[datetime, datetime]:
        """Return the fixed trace-time interval used for accounting.

        Historical execution start is absent from the trace, so the interval
        is reconstructed as submission through submission plus the historical
        wall-clock duration.
        """
        start = job.submit_time
        return start, start + timedelta(seconds=max(0.0, float(job.wallclock)))

    @staticmethod
    def _weighted_hourly_average(
        start: datetime,
        end: datetime,
        sample_hour: Any,
    ) -> float:
        """Average hourly provider values, weighted by interval overlap."""
        if end <= start:
            return float(sample_hour(start))
        weighted_sum = 0.0
        total_seconds = 0.0
        cursor = start
        while cursor < end:
            next_hour = cursor.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)
            segment_end = min(end, next_hour)
            overlap_seconds = (segment_end - cursor).total_seconds()
            weighted_sum += float(sample_hour(cursor)) * overlap_seconds
            total_seconds += overlap_seconds
            cursor = segment_end
        return weighted_sum / total_seconds

    def _point_timestamp(self, site: Site, job: Job) -> datetime:
        start_time = job.start_time or self.current_time
        if self.environmental_time_mode == "submission":
            return job.submit_time
        if self.environmental_time_mode == "execution_start":
            return start_time
        if self.environmental_time_mode == "execution_midpoint":
            runtime_min = int(job.assigned_runtime_min)
            if runtime_min <= 0:
                _, _, runtime_min = self.derive_job_runtime_for_site(job, site)
            return start_time + timedelta(minutes=max(1, runtime_min) / 2.0)
        raise RuntimeError("point timestamp requested for interval-average mode")

    def ci_for_job(self, site: Site, job: Job) -> float:
        def sample(timestamp: datetime) -> float:
            return self.ci_provider.get_ci(
                site_name=site.name,
                midpoint_ts=timestamp,
                latitude=site.latitude,
                longitude=site.longitude,
            )

        if self.environmental_time_mode == "historical_interval_average":
            start, end = self._historical_interval(job)
            job.environmental_interval_start = start
            job.environmental_interval_end = end
            job.environmental_time_mode = self.environmental_time_mode
            return self._weighted_hourly_average(start, end, sample)

        timestamp = self._point_timestamp(site, job)
        job.environmental_interval_start = timestamp
        job.environmental_interval_end = timestamp
        job.environmental_time_mode = self.environmental_time_mode
        return sample(timestamp)

    def hydric_impact_for_job(self, site: Site, job: Job) -> float:
        """Return site/time water-scarcity intensity in stress-L/kWh.

        A provider is optional at the simulator level so carbon-only callers
        can continue to run; in that case water-impact accounting remains zero.
        """
        if self.hydric_impact_provider is None:
            return 0.0
        def sample(timestamp: datetime) -> float:
            return self.hydric_impact_provider.get_hydric_impact(
                site_name=site.name,
                impact_ts=timestamp,
                latitude=site.latitude,
                longitude=site.longitude,
            )

        if self.environmental_time_mode == "historical_interval_average":
            start, end = self._historical_interval(job)
            return self._weighted_hourly_average(start, end, sample)
        return sample(self._point_timestamp(site, job))

    def derive_job_runtime_for_site(self, job: Job, site: Site) -> Tuple[float, float, int]:
        # perf_hs06 is a site speed factor:
        # higher perf => less CPU seconds needed for the same normalized workload.
        perf = float(site.perf_hs06)
        if perf <= 0.0:
            return 0.0, 0.0, 0
        cpu_seconds_sim = float(job.norm_cpu_seconds) / perf

        # Derive wallclock from job metadata and site performance.
        wallclock_seconds_sim = (float(job.wallclock) * float(job.cpu_norm_factor)) / perf
        runtime_min_sim = max(1, int(math.ceil(wallclock_seconds_sim / 60.0)))
        return cpu_seconds_sim, wallclock_seconds_sim, runtime_min_sim

    def compute_energy_kwh(self, job: Job, site: Site) -> float:
        cpu_seconds, wallclock_seconds, _ = self.derive_job_runtime_for_site(job, site)
        total_cores = int(site.avg_total_cores)
        cores_used = int(job.cores_used)
        tdp = float(site.avg_tdp_w)
        if wallclock_seconds <= 0.0 or total_cores <= 0:
            return 0.0

        f = max(0.0, min(1.0, float(self.idle_consumption_factor)))
        effective_time_s = (1.0 - f) * cpu_seconds + f * wallclock_seconds
        core_fraction = float(cores_used) / float(total_cores)
        energy_joule = effective_time_s * core_fraction * tdp
        return energy_joule / 3_600_000.0

    def step_match(self) -> None:
        # Waiting queue is already in stable submit order.
        waiting = list(self.waiting_queue)
        self.waiting_history.append((self.current_time, len(waiting)))
        # Ask the policy which site should take how many jobs now.
        submissions = self.policy.schedule(
            waiting,
            self.sites,
            current_time=self.current_time,
        )
        self.submissions_history.append((self.current_time, {site: k for site, k in submissions}))
        if submissions:
            logger.info("match t=%s waiting=%d submissions=%s", self.current_time.isoformat(),
                        len(waiting), submissions)

        for site_name, k in submissions:
            site = self.sites[site_name]
            quota = k

            # Start up to "quota" jobs on this site.
            for _ in range(quota):
                if not self.waiting_queue:
                    break
                picked = self.waiting_queue.popleft()
                # Mark job as running and stamp start metadata.
                picked.status = "running"
                picked.start_time = self.current_time
                picked.site = site.name
                picked.site_rank_at_start = self.policy.site_rank(site.name)
                if self.policy.delay_applies_to_rank(picked.site_rank_at_start):
                    picked.spillover_delay_threshold_min = (
                        self.policy.spillover_delay_minutes
                    )
                # Compute runtime and energy numbers for this site/job pair.
                cpu_seconds_sim, wallclock_seconds_sim, runtime_min_sim = self.derive_job_runtime_for_site(picked,
                                                                                                           site)
                picked.assigned_cpu_seconds = cpu_seconds_sim
                picked.assigned_wallclock_seconds = wallclock_seconds_sim
                picked.assigned_runtime_min = runtime_min_sim
                picked.remaining_min = runtime_min_sim
                total_kwh = self.compute_energy_kwh(picked, site)
                picked.total_energy_kwh = total_kwh
                picked.assigned_ci_gco2_per_kwh = self.ci_for_job(site, picked)
                picked.carbon_kg = (picked.total_energy_kwh * picked.assigned_ci_gco2_per_kwh) / 1000.0
                picked.assigned_hi_stress_l_per_kwh = self.hydric_impact_for_job(site, picked)
                if self.hydric_impact_provider is not None:
                    if site.pue is None or float(site.pue) <= 0.0:
                        raise ValueError(f"Valid PUE required for hydric impact at site={site.name}")
                    picked.water_impact_stress_l = (
                        picked.total_energy_kwh
                        * float(site.pue)
                        * picked.assigned_hi_stress_l_per_kwh
                    )
                # Put job into site's running list and remove it from waiting list.
                site.running_jobs.append(picked)
                logger.debug("start job=%s site=%s rt=%d", picked.job_id,
                             site.name, picked.assigned_runtime_min)

    def step_execute(self) -> None:
        # Run one simulation minute for every running job.
        finished = 0
        for site in self.sites.values():
            for job in list(site.running_jobs):
                # Job worked for one more minute.
                job.remaining_min -= 1
                if job.remaining_min > 0:
                    continue
                # Job is finished now: mark done and move it to completed list.
                job.status = "done"
                job.finish_time = self.current_time + self.tick
                self.done_jobs.append(job)
                self.done_count += 1
                site.running_jobs.remove(job)
                finished += 1
        if finished:
            logger.info("finish t=%s jobs=%d", self.current_time.isoformat(), finished)

    def step(self) -> None:
        logger.debug("step t=%s active=%d", self.current_time.isoformat(), self.active_jobs())
        # STEP 1: Release new jobs (pending -> waiting).
        self.release_jobs()
        # STEP 2: Match waiting jobs to sites and start them.
        self.step_match()
        # STEP 3: Execute one tick and finish jobs that reached 0 time left.
        self.step_execute()
        # Snapshot running jobs per site for reporting.
        self.running_history.append((self.current_time, self.running_jobs_by_site()))
        # STEP 4: Move clock forward by one tick (usually 1 minute).
        self.current_time += self.tick

    def done(self) -> bool:
        all_jobs_done = self.done_count == self.total_jobs
        no_active = self.active_jobs() == 0
        return all_jobs_done and no_active
