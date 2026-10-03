import unittest
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Job, Site
from policy import ReplayCarbonPolicy
from simulator import ReplaySimulator


class DummyCIProvider:
    def __init__(self):
        self.calls = []

    def get_ci(self, site_name, midpoint_ts, latitude, longitude):
        self.calls.append((site_name, midpoint_ts, latitude, longitude))
        return 100.0


def make_site(name: str, score: float) -> Site:
    return Site(
        name=name,
        max_running_jobs=1,
        green=score,
        greenhydric=score,
        latitude=0.0,
        longitude=0.0,
        avg_tdp_w=120.0,
        avg_total_cores=12,
        perf_hs06=1.0,
    )


class DelayPolicyIntegrationTests(unittest.TestCase):
    def test_job_spills_only_after_waiting_threshold(self):
        submitted = datetime(2026, 1, 1, 0, 0, 0)
        long_job = Job(
            job_id="J1",
            submit_time=submitted,
            norm_cpu_seconds=600.0,
            wallclock=600.0,
            cpu_norm_factor=1.0,
        )
        waiting_job = Job(
            job_id="J2",
            submit_time=submitted,
            norm_cpu_seconds=60.0,
            wallclock=60.0,
            cpu_norm_factor=1.0,
        )
        policy = ReplayCarbonPolicy(
            green=1,
            hydric_enabled=True,
            spillover_delay_minutes=2,
            preferred_site_count=1,
        )
        provider = DummyCIProvider()
        sim = ReplaySimulator(
            sites={
                "BEST": make_site("BEST", 10.0),
                "LOWER": make_site("LOWER", 1.0),
            },
            jobs=[long_job, waiting_job],
            tick_minutes=1,
            policy=policy,
            ci_provider=provider,
            environmental_time_mode="execution_midpoint",
        )

        sim.step()  # t=0: J1 starts at BEST; J2 is too fresh to spill.
        self.assertEqual("waiting", waiting_job.status)
        sim.step()  # t=1: J2 has waited only one minute.
        self.assertEqual("waiting", waiting_job.status)
        sim.step()  # t=2: threshold reached; J2 may start at LOWER.

        self.assertEqual("LOWER", waiting_job.site)
        self.assertEqual(submitted.replace(minute=2), waiting_job.start_time)
        self.assertEqual(2, waiting_job.site_rank_at_start)
        self.assertEqual(2, waiting_job.spillover_delay_threshold_min)
        # The one-minute job's execution midpoint is 30 seconds after start.
        lower_call = next(call for call in provider.calls if call[0] == "LOWER")
        self.assertEqual(submitted.replace(minute=2, second=30), lower_call[1])


if __name__ == "__main__":
    unittest.main()
