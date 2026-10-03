import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

# Allow running tests from any cwd (IDE or CLI).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci_provider import MidpointCIProvider
from csv_io import load_jobs
from models import Job, Site
from policy import ReplayCarbonPolicy
from simulator import ReplaySimulator


def make_site(name: str, max_running_jobs: int = 2, green: float = 0.5) -> Site:
    return Site(
        name=name,
        max_running_jobs=max_running_jobs,
        green=green,
        latitude=52.0,
        longitude=4.0,
        avg_tdp_w=180.0,
        avg_total_cores=24,
        perf_hs06=1.0,
    )


def make_job(job_id: str, submit: datetime, norm_cpu_seconds: float = 120.0) -> Job:
    return Job(
        job_id=job_id,
        submit_time=submit,
        norm_cpu_seconds=norm_cpu_seconds,
        cores_used=1,
    )


class _Resp:
    def __init__(self, ci: float):
        self._ci = ci

    def raise_for_status(self):
        return None

    def json(self):
        return {"ci_gco2_per_kwh": self._ci}


class DummyCIProvider:
    def get_ci(self, site_name, midpoint_ts, latitude, longitude):
        return 200.0


class AdditionalTests(unittest.TestCase):
    def test_load_jobs_reads_wallclock_and_cpu_norm_factor(self):
        csv_content = (
            "job_id,submit_time,norm_cpu_seconds,cores_used,wallclock,CPUNormFactor\n"
            "J1,2026-01-01 00:00:00,120.0,2,7.5,4.2\n"
        )
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "jobs.csv"
            csv_path.write_text(csv_content, encoding="utf-8")
            jobs = load_jobs(csv_path)

        self.assertEqual(1, len(jobs))
        self.assertEqual(7.5, jobs[0].wallclock)
        self.assertEqual(4.2, jobs[0].cpu_norm_factor)

    def test_load_jobs_reads_dirac_parameter_names(self):
        csv_content = (
            "job_id,submit_time,NormCPUTime(s),cores_used,WallClockTime,CPUNormalizationFactor\n"
            "J1,2026-01-01 00:00:00,120.0,2,61.024,33.9\n"
        )
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "jobs.csv"
            csv_path.write_text(csv_content, encoding="utf-8")
            jobs = load_jobs(csv_path)

        self.assertEqual(1, len(jobs))
        self.assertEqual(61.024, jobs[0].wallclock)
        self.assertEqual(33.9, jobs[0].cpu_norm_factor)

    def test_schedule_spreads_across_sites_by_e_and_capacity(self):
        policy = ReplayCarbonPolicy()
        sites = {
            "S1": make_site("S1", max_running_jobs=1, green=0.1),
            "S2": make_site("S2", max_running_jobs=2, green=0.2),
            "S3": make_site("S3", max_running_jobs=5, green=0.9),
        }
        jobs = [make_job(f"J{i}", datetime(2026, 1, 1, 0, 0, 0)) for i in range(1, 5)]

        submissions = policy.schedule(jobs, sites)
        self.assertEqual([("S3", 4)], submissions)

    def test_step_match_assigns_fifo_with_site_quotas(self):
        sites = {
            "S1": make_site("S1", max_running_jobs=1, green=0.1),
            "S2": make_site("S2", max_running_jobs=2, green=0.8),
        }
        jobs = [
            make_job("J2", datetime(2026, 1, 1, 0, 0, 0)),
            make_job("J1", datetime(2026, 1, 1, 0, 0, 0)),
            make_job("J3", datetime(2026, 1, 1, 0, 0, 0)),
        ]

        sim = ReplaySimulator(sites=sites, jobs=jobs, tick_minutes=1, ci_provider=DummyCIProvider())
        sim.current_time = datetime(2026, 1, 1, 0, 0, 0)
        sim.release_jobs()
        sim.step_match()

        self.assertEqual("S2", next(j for j in jobs if j.job_id == "J1").site)
        self.assertEqual("S2", next(j for j in jobs if j.job_id == "J2").site)
        self.assertEqual("S1", next(j for j in jobs if j.job_id == "J3").site)
        self.assertEqual(1, len(sites["S1"].running_jobs))
        self.assertEqual(2, len(sites["S2"].running_jobs))

    def test_done_is_false_until_jobs_are_done(self):
        site = make_site("S1", max_running_jobs=1, green=0.1)
        job = make_job("J1", datetime(2026, 1, 1, 0, 0, 0), norm_cpu_seconds=30.0)
        sim = ReplaySimulator(sites={"S1": site}, jobs=[job], tick_minutes=1, ci_provider=DummyCIProvider())

        self.assertFalse(sim.done())  # job is still pending

        sim.step()  # release + match + execute one minute
        self.assertTrue(sim.done())   # job completed and no active jobs

    def test_ci_provider_keeps_multiple_buckets_until_ttl(self):
        provider = MidpointCIProvider(token="t", kpi_api_base="https://kpi.example")

        with patch("ci_provider.requests.post", side_effect=[_Resp(111.0), _Resp(222.0)]):
            provider.get_ci("SARA", datetime(2026, 1, 1, 10, 5, 0), 52.0, 4.0)   # 10:00 bucket
            provider.get_ci("SARA", datetime(2026, 1, 1, 11, 5, 0), 52.0, 4.0)   # 11:00 bucket

        old_key = ("SARA", datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc))
        new_key = ("SARA", datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc))

        self.assertIn(old_key, provider.cache)
        self.assertIn(new_key, provider.cache)
        self.assertEqual(2, len(provider.cache))


if __name__ == "__main__":
    unittest.main()
