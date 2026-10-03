import unittest
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Allow running tests from any cwd (IDE or CLI).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Job, Site
from policy import ReplayCarbonPolicy


def make_site(
    name: str,
    max_running_jobs: int = 2,
    green: float = 0.5,
    greenhydric: float = 0.0,
) -> Site:
    return Site(
        name=name,
        max_running_jobs=max_running_jobs,
        green=green,
        latitude=0.0,
        longitude=0.0,
        avg_tdp_w=150.0,
        avg_total_cores=12,
        perf_hs06=1.0,
        greenhydric=greenhydric,
    )


def make_job(job_id: str) -> Job:
    return Job(
        job_id=job_id,
        submit_time=datetime(2026, 1, 1, 0, 0, 0),
        norm_cpu_seconds=60,
        cores_used=1,
    )


class PolicyTests(unittest.TestCase):
    def test_unmet_jobs_shared_site_capacity_counted_once(self):
        policy = ReplayCarbonPolicy()
        sara = make_site("SARA", max_running_jobs=1)
        sites = {"SARA": sara}
        jobs = [
            make_job("J1"),
            make_job("J2"),
        ]

        unmet = policy.unmet_jobs(jobs, sites)
        self.assertEqual(1, len(unmet))
        self.assertEqual("J2", unmet[0].job_id)

    def test_schedule_places_on_highest_greenscore_site(self):
        policy = ReplayCarbonPolicy()
        sites = {
            "SARA": make_site("SARA", max_running_jobs=1, green=0.1),
            "NIKHEF": make_site("NIKHEF", max_running_jobs=1, green=0.9),
        }
        jobs = [make_job("J1")]

        submissions = policy.schedule(jobs, sites)
        self.assertEqual([("NIKHEF", 1)], submissions)

    def test_hydric_schedule_places_on_highest_greenhydric_site(self):
        policy = ReplayCarbonPolicy(hydric_enabled=True)
        sites = {
            "SARA": make_site("SARA", max_running_jobs=1, green=0.9, greenhydric=0.1),
            "NIKHEF": make_site("NIKHEF", max_running_jobs=1, green=0.1, greenhydric=0.9),
        }

        submissions = policy.schedule([make_job("J1")], sites)

        self.assertEqual([("NIKHEF", 1)], submissions)

    def test_spillover_delay_keeps_fresh_job_off_lower_ranked_site(self):
        now = datetime(2026, 1, 1, 1, 0, 0)
        policy = ReplayCarbonPolicy(
            green=1,
            hydric_enabled=True,
            spillover_delay_minutes=60,
        )
        sites = {
            "BEST": make_site("BEST", max_running_jobs=0, greenhydric=10.0),
            "LOWER": make_site("LOWER", max_running_jobs=2, greenhydric=1.0),
        }
        fresh = make_job("J1")
        fresh.submit_time = now - timedelta(minutes=59)

        submissions = policy.schedule([fresh], sites, current_time=now)

        self.assertEqual([], submissions)

    def test_spillover_delay_releases_job_at_exact_threshold(self):
        now = datetime(2026, 1, 1, 1, 0, 0)
        policy = ReplayCarbonPolicy(
            green=1,
            hydric_enabled=True,
            spillover_delay_minutes=60,
        )
        sites = {
            "BEST": make_site("BEST", max_running_jobs=0, greenhydric=10.0),
            "LOWER": make_site("LOWER", max_running_jobs=2, greenhydric=1.0),
        }
        old_enough = make_job("J1")
        old_enough.submit_time = now - timedelta(minutes=60)

        submissions = policy.schedule([old_enough], sites, current_time=now)

        self.assertEqual([("LOWER", 1)], submissions)

    def test_random_policy_does_not_apply_green_spillover_delay(self):
        now = datetime(2026, 1, 1, 1, 0, 0)
        policy = ReplayCarbonPolicy(
            green=0,
            spillover_delay_minutes=60,
        )
        sites = {"SITE": make_site("SITE", max_running_jobs=1)}
        fresh = make_job("J1")
        fresh.submit_time = now

        submissions = policy.schedule([fresh], sites, current_time=now)

        self.assertEqual([("SITE", 1)], submissions)

    def test_delayed_policy_requires_current_time(self):
        policy = ReplayCarbonPolicy(green=1, spillover_delay_minutes=60)
        sites = {"SITE": make_site("SITE", max_running_jobs=1)}

        with self.assertRaisesRegex(ValueError, "current_time"):
            policy.schedule([make_job("J1")], sites)


if __name__ == "__main__":
    unittest.main()
