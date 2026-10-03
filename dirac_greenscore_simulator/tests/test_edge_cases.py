import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from requests.exceptions import HTTPError
import sys

# Allow running tests from any cwd (IDE or CLI).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci_provider import MidpointCIProvider
from models import Job, Site
from policy import ReplayCarbonPolicy
from simulator import ReplaySimulator


def make_site(name: str, cores: int = 24, tdp: float = 180.0) -> Site:
    return Site(
        name=name,
        max_running_jobs=2,
        green=0.5,
        latitude=52.0,
        longitude=4.0,
        avg_tdp_w=tdp,
        avg_total_cores=cores,
        perf_hs06=1.0,
    )


def make_job(job_id: str, submit: datetime, norm_cpu_seconds: float = 900.0) -> Job:
    return Job(
        job_id=job_id,
        submit_time=submit,
        norm_cpu_seconds=norm_cpu_seconds,
        cores_used=1,
        wallclock=int(norm_cpu_seconds),
        cpu_norm_factor=1,
    )


class _Resp:
    def __init__(self, ci):
        self._ci = ci

    def raise_for_status(self):
        return None

    def json(self):
        return {"ci_gco2_per_kwh": self._ci}

class _TokenResp:
    def __init__(self, payload, raise_http=False):
        self._payload = payload
        self._raise_http = raise_http

    def raise_for_status(self):
        if self._raise_http:
            raise HTTPError("auth failed")
        return None

    def json(self):
        return self._payload


class DummyCIProvider:
    def __init__(self, ci=100.0):
        self.ci = ci
        self.calls = []

    def get_ci(self, site_name, midpoint_ts, latitude, longitude):
        self.calls.append((site_name, midpoint_ts, latitude, longitude))
        return self.ci


class EdgeCaseTests(unittest.TestCase):
    def test_ci_provider_hour_bucket_cache(self):
        provider = MidpointCIProvider(token="t", kpi_api_base="https://kpi.example")

        with patch("ci_provider.requests.post", side_effect=[_Resp(111.0), _Resp(222.0)]) as post:
            ci1 = provider.get_ci("SARA", datetime(2026, 1, 1, 10, 5, 0), 52.0, 4.0)
            ci2 = provider.get_ci("SARA", datetime(2026, 1, 1, 10, 25, 0), 52.0, 4.0)
            ci3 = provider.get_ci("SARA", datetime(2026, 1, 1, 11, 5, 0), 52.0, 4.0)

        self.assertEqual(111.0, ci1)
        self.assertEqual(111.0, ci2)  # same hour bucket -> cache hit
        self.assertEqual(222.0, ci3)  # next hour bucket -> new HTTP call
        self.assertEqual(2, post.call_count)

    def test_ci_provider_missing_coords_reuses_cached_ci_without_http(self):
        provider = MidpointCIProvider(token="t", kpi_api_base="https://kpi.example", fallback_ci=333.0)
        provider._cache_set("SARA", provider._hour_bucket(datetime(2026, 1, 1, 9, 0, 0)), 222.0)

        with patch("ci_provider.requests.post") as post:
            ci1 = provider.get_ci("SARA", datetime(2026, 1, 1, 10, 5, 0), None, None)
            ci2 = provider.get_ci("SARA", datetime(2026, 1, 1, 10, 10, 0), None, None)

        self.assertEqual(222.0, ci1)
        self.assertEqual(222.0, ci2)
        self.assertEqual(0, post.call_count)

    def test_ci_provider_request_error_without_cache_raises(self):
        provider = MidpointCIProvider(token="t", kpi_api_base="https://kpi.example", fallback_ci=333.0)
        with patch("ci_provider.requests.post", side_effect=HTTPError("kpi down")):
            with self.assertRaises(RuntimeError):
                provider.get_ci("SARA", datetime(2026, 1, 1, 10, 5, 0), 52.0, 4.0)

    def test_get_token_uses_get_only(self):
        provider = MidpointCIProvider(
            cim_api_base="https://cim.example",
            email="user@example.org",
            password="secret",
        )

        with patch(
            "ci_provider.requests.get",
            return_value=_TokenResp({"token": "tok-get"}),
        ) as get, patch("ci_provider.requests.post") as post:
            token = provider._get_token()

        self.assertEqual("tok-get", token)
        self.assertEqual(1, get.call_count)
        self.assertEqual(0, post.call_count)

    def test_get_token_returns_none_on_get_http_error(self):
        provider = MidpointCIProvider(
            cim_api_base="https://cim.example",
            email="user@example.org",
            password="secret",
        )

        with patch(
            "ci_provider.requests.get",
            return_value=_TokenResp({}, raise_http=True),
        ) as get:
            token = provider._get_token()

        self.assertIsNone(token)
        self.assertEqual(1, get.call_count)

    def test_policy_schedule_places_job_even_without_matching_tags(self):
        policy = ReplayCarbonPolicy()
        sites = {"SARA": make_site("SARA")}
        jobs = [make_job("J1", datetime(2026, 1, 1, 0, 0, 0))]

        submissions = policy.schedule(jobs, sites)
        self.assertEqual([("SARA", 1)], submissions)

    def test_simulator_ci_for_job_uses_execution_midpoint_provider(self):
        site = make_site("SARA")
        job = make_job("J1", datetime(2026, 1, 1, 12, 10, 0), norm_cpu_seconds=1200.0)
        provider = DummyCIProvider(ci=123.0)
        sim = ReplaySimulator(
            sites={"SARA": site},
            jobs=[job],
            tick_minutes=1,
            ci_provider=provider,
            environmental_time_mode="execution_midpoint",
        )

        ci = sim.ci_for_job(site, job)
        self.assertEqual(123.0, ci)
        self.assertEqual(1, len(provider.calls))
        _, ci_ts, _, _ = provider.calls[0]
        # Runtime is 20 minutes, so the execution midpoint is 10 minutes later.
        self.assertEqual(datetime(2026, 1, 1, 12, 20, 0), ci_ts)

    def test_compute_energy_guard_conditions(self):
        site_zero_cores = make_site("SARA", cores=0)
        job = make_job("J1", datetime(2026, 1, 1, 0, 0, 0))
        sim = ReplaySimulator(sites={"SARA": site_zero_cores}, jobs=[job], tick_minutes=1, ci_provider=DummyCIProvider())
        self.assertEqual(0.0, sim.compute_energy_kwh(job, site_zero_cores))

        site_ok = make_site("SARA", cores=24)
        job_zero_norm = make_job("J2", datetime(2026, 1, 1, 0, 0, 0), norm_cpu_seconds=0.0)
        self.assertEqual(0.0, sim.compute_energy_kwh(job_zero_norm, site_ok))

        site_zero_perf = make_site("SARA", cores=24)
        site_zero_perf.perf_hs06 = 0.0
        self.assertEqual(0.0, sim.compute_energy_kwh(job, site_zero_perf))

    def test_release_jobs_boundary_exact_current_time(self):
        site = make_site("SARA")
        now = datetime(2026, 1, 1, 0, 0, 0)
        j_now = make_job("J_now", now)
        j_later = make_job("J_later", datetime(2026, 1, 1, 0, 1, 0))
        sim = ReplaySimulator(sites={"SARA": site}, jobs=[j_now, j_later], tick_minutes=1, ci_provider=DummyCIProvider())
        sim.current_time = now

        sim.release_jobs()
        self.assertEqual("waiting", j_now.status)
        self.assertEqual("pending", j_later.status)

    def test_runtime_derivation_uses_job_wallclock_and_site_perf(self):
        site = make_site("SARA")
        site.perf_hs06 = 2.0
        job = make_job("J1", datetime(2026, 1, 1, 0, 0, 0), norm_cpu_seconds=600.0)
        sim = ReplaySimulator(sites={"SARA": site}, jobs=[job], tick_minutes=1, ci_provider=DummyCIProvider())

        cpu_s, wall_s, runtime_m = sim.derive_job_runtime_for_site(job, site)
        self.assertEqual(300.0, cpu_s)   # 600 / 2.0
        self.assertEqual(300.0, wall_s)  # 600 * 1 / 2.0
        self.assertEqual(5, runtime_m)   # ceil(300/60) = 5


if __name__ == "__main__":
    unittest.main()
