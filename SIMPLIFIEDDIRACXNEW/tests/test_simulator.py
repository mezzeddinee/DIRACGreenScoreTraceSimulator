import unittest
from datetime import datetime
from pathlib import Path
import sys

# Allow running tests from any cwd (IDE or CLI).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Job, Site
from simulator import ReplaySimulator


def make_site(name: str, max_running_jobs: int = 2) -> Site:
    return Site(
        name=name,
        max_running_jobs=max_running_jobs,
        green=0.5,
        latitude=52.0,
        longitude=4.0,
        avg_tdp_w=180.0,
        avg_total_cores=24,
        perf_hs06=1.0,
    )


def make_job(job_id: str, submit: datetime) -> Job:
    return Job(
        job_id=job_id,
        submit_time=submit,
        norm_cpu_seconds=90.0,
        cores_used=1,
        wallclock=90,
        cpu_norm_factor=1,
    )


class DummyCIProvider:
    def __init__(self):
        self.calls = []

    def get_ci(self, site_name, midpoint_ts, latitude, longitude):
        self.calls.append((site_name, midpoint_ts, latitude, longitude))
        return 200.0


class HourlyCIProvider(DummyCIProvider):
    def get_ci(self, site_name, midpoint_ts, latitude, longitude):
        self.calls.append((site_name, midpoint_ts, latitude, longitude))
        return {10: 100.0, 11: 200.0, 12: 300.0}[midpoint_ts.hour]


class HourlyHydricProvider:
    def __init__(self):
        self.calls = []

    def get_hydric_impact(self, site_name, impact_ts, latitude, longitude):
        self.calls.append((site_name, impact_ts, latitude, longitude))
        return {10: 2.0, 11: 4.0, 12: 6.0}[impact_ts.hour]


class SimulatorTests(unittest.TestCase):
    def test_compute_energy_kwh_matches_formula(self):
        site = make_site("SARA")
        job = make_job("J1", datetime(2026, 1, 1, 0, 0, 0))
        sim = ReplaySimulator(sites={"SARA": site}, jobs=[job], tick_minutes=1, ci_provider=DummyCIProvider())

        energy = sim.compute_energy_kwh(job, site)
        # derived wallclock=job.wallclock*job.cpu_norm_factor/perf=90
        # E=((1-f)*90 + f*90) * (1/24) * 180 / 3_600_000 = 0.0001875
        self.assertAlmostEqual(0.0001875, energy, places=9)

    def test_ci_for_job_uses_execution_midpoint_with_provider(self):
        site = make_site("SARA")
        job = make_job("J1", datetime(2026, 1, 1, 12, 0, 0))
        provider = DummyCIProvider()
        sim = ReplaySimulator(
            sites={"SARA": site},
            jobs=[job],
            tick_minutes=1,
            ci_provider=provider,
            environmental_time_mode="execution_midpoint",
        )

        ci = sim.ci_for_job(site, job)
        self.assertEqual(200.0, ci)
        self.assertEqual(1, len(provider.calls))
        site_name, ci_ts, lat, lon = provider.calls[0]
        self.assertEqual("SARA", site_name)
        # Runtime is ceil(90 / 60) = 2 minutes, so midpoint is +1 minute.
        self.assertEqual(datetime(2026, 1, 1, 12, 1, 0), ci_ts)
        self.assertEqual(52.0, lat)
        self.assertEqual(4.0, lon)

    def test_submission_ci_timestamp_mode_preserves_legacy_behavior(self):
        site = make_site("SARA")
        submitted = datetime(2026, 1, 1, 12, 0, 0)
        job = make_job("J1", submitted)
        provider = DummyCIProvider()
        sim = ReplaySimulator(
            sites={"SARA": site},
            jobs=[job],
            tick_minutes=1,
            ci_provider=provider,
            ci_timestamp_mode="submission",
        )

        sim.ci_for_job(site, job)

        self.assertEqual(submitted, provider.calls[0][1])

    def test_default_uses_weighted_historical_interval_for_both_signals(self):
        site = make_site("SARA")
        submitted = datetime(2026, 1, 1, 10, 30, 0)
        job = Job(
            job_id="J1",
            submit_time=submitted,
            norm_cpu_seconds=7200.0,
            cores_used=1,
            wallclock=7200.0,
            cpu_norm_factor=1.0,
        )
        job.start_time = datetime(2026, 1, 2, 20, 0, 0)
        ci_provider = HourlyCIProvider()
        hydric_provider = HourlyHydricProvider()
        sim = ReplaySimulator(
            sites={"SARA": site},
            jobs=[job],
            ci_provider=ci_provider,
            hydric_impact_provider=hydric_provider,
        )

        ci = sim.ci_for_job(site, job)
        wi = sim.hydric_impact_for_job(site, job)

        # Historical interval is 10:30--12:30: 0.5h, 1h, 0.5h.
        self.assertAlmostEqual(200.0, ci)
        self.assertAlmostEqual(4.0, wi)
        expected_calls = [
            datetime(2026, 1, 1, 10, 30),
            datetime(2026, 1, 1, 11, 0),
            datetime(2026, 1, 1, 12, 0),
        ]
        self.assertEqual(expected_calls, [call[1] for call in ci_provider.calls])
        self.assertEqual(expected_calls, [call[1] for call in hydric_provider.calls])
        self.assertEqual("historical_interval_average", job.environmental_time_mode)
        self.assertEqual(submitted, job.environmental_interval_start)
        self.assertEqual(datetime(2026, 1, 1, 12, 30), job.environmental_interval_end)

    def test_step_match_and_execute_completes_job(self):
        site = make_site("SARA", max_running_jobs=1)
        job = make_job("J1", datetime(2026, 1, 1, 0, 0, 0))

        sim = ReplaySimulator(
            sites={"SARA": site},
            jobs=[job],
            tick_minutes=1,
            ci_provider=DummyCIProvider(),
        )
        sim.current_time = datetime(2026, 1, 1, 0, 0, 0)

        sim.release_jobs()
        sim.step_match()
        self.assertEqual("running", job.status)
        self.assertEqual("SARA", job.site)
        self.assertEqual(1, len(site.running_jobs))

        sim.step_execute()
        self.assertEqual("running", job.status)
        self.assertGreater(job.carbon_kg, 0.0)

        sim.step_execute()
        self.assertEqual("done", job.status)
        self.assertEqual(0, len(site.running_jobs))


if __name__ == "__main__":
    unittest.main()
