import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci_provider import MidpointCIProvider
from csv_io import load_sites
from hydric_impact_provider import WattNetHydricImpactProvider
from models import Job, Site
from simulator import ReplaySimulator


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _CIProvider:
    def get_ci(self, site_name, midpoint_ts, latitude, longitude):
        return 200.0


class _HydricProvider:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def get_hydric_impact(self, site_name, impact_ts, latitude, longitude):
        self.calls.append((site_name, impact_ts, latitude, longitude))
        return self.value


class HydricImpactTests(unittest.TestCase):
    def test_provider_queries_wattnet_and_caches_hour_bucket(self):
        provider = WattNetHydricImpactProvider(
            token="token",
            api_base="https://wattnet.example/v1",
        )
        response = _Response(
            [
                {
                    "value": 3.5,
                    "valid": True,
                    "zone": "NL",
                    "zone_status": "complete",
                }
            ]
        )

        with patch.object(provider.session, "get", return_value=response) as get:
            first = provider.get_hydric_impact(
                "SARA",
                datetime(2026, 6, 1, 10, 5),
                52.0,
                4.0,
            )
            second = provider.get_hydric_impact(
                "SARA",
                datetime(2026, 6, 1, 10, 45),
                52.0,
                4.0,
            )

        self.assertEqual(3.5, first)
        self.assertEqual(3.5, second)
        self.assertEqual(1, get.call_count)
        _, kwargs = get.call_args
        self.assertEqual("water", kwargs["params"]["impact_type"])
        self.assertEqual("operational", kwargs["params"]["scope"])
        self.assertTrue(kwargs["params"]["aggregate"])
        self.assertTrue(kwargs["params"]["use_global"])
        self.assertEqual("2026-06-01T10:00:00Z", kwargs["params"]["start"])
        self.assertEqual("2026-06-01T11:00:00Z", kwargs["params"]["end"])

    def test_cim_pue_lookup_is_cached_by_site(self):
        provider = MidpointCIProvider(
            token="token",
            kpi_api_base="https://kpi.example/v1",
        )
        response = _Response(
            {
                "site_name": "SARA",
                "pue": 1.21,
                "location": {"latitude": 52.0, "longitude": 4.0},
                "source": "gocdb",
            }
        )

        with patch("ci_provider.requests.post", return_value=response) as post:
            first = provider.get_pue("SARA", 52.0, 4.0)
            second = provider.get_pue("SARA", 52.0, 4.0)

        self.assertEqual(1.21, first)
        self.assertEqual(1.21, second)
        self.assertEqual(1, post.call_count)
        _, kwargs = post.call_args
        self.assertEqual({"site_name": "SARA"}, kwargs["json"])

    def test_job_water_impact_uses_energy_pue_and_midpoint_hi(self):
        site = Site(
            name="SARA",
            max_running_jobs=1,
            green=1.0,
            latitude=52.0,
            longitude=4.0,
            avg_tdp_w=180.0,
            avg_total_cores=24,
            perf_hs06=1.0,
            pue=2.0,
        )
        job = Job(
            job_id="J1",
            submit_time=datetime(2026, 6, 1, 0, 0),
            norm_cpu_seconds=120.0,
            cores_used=1,
            wallclock=120.0,
            cpu_norm_factor=1.0,
        )
        hydric = _HydricProvider(10.0)
        sim = ReplaySimulator(
            sites={"SARA": site},
            jobs=[job],
            ci_provider=_CIProvider(),
            hydric_impact_provider=hydric,
            environmental_time_mode="execution_midpoint",
        )

        sim.release_jobs()
        sim.step_match()

        self.assertAlmostEqual(0.00025, job.total_energy_kwh)
        self.assertEqual(10.0, job.assigned_hi_stress_l_per_kwh)
        self.assertAlmostEqual(0.005, job.water_impact_stress_l)
        self.assertEqual(1, len(hydric.calls))
        self.assertEqual(datetime(2026, 6, 1, 0, 1), hydric.calls[0][1])

    def test_site_loader_reads_pue(self):
        csv_content = (
            "site,max_running_jobs,green,latitude,longitude,"
            "avg_tdp_w,avg_total_cores,perf_hs06,pue\n"
            "SARA,2,1.0,52.0,4.0,180,24,1.0,1.21\n"
        )
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sites.csv"
            path.write_text(csv_content, encoding="utf-8")
            sites = load_sites(path)

        self.assertEqual(1.21, sites["SARA"].pue)

    def test_site_loader_reads_greenhydric_and_defaults_to_zero(self):
        with_greenhydric = (
            "site,max_running_jobs,green,greenhydric,latitude,longitude,"
            "avg_tdp_w,avg_total_cores,perf_hs06\n"
            "SARA,2,1.0,4.2,52.0,4.0,180,24,1.0\n"
        )
        without_greenhydric = (
            "site,max_running_jobs,green,latitude,longitude,"
            "avg_tdp_w,avg_total_cores,perf_hs06\n"
            "SARA,2,1.0,52.0,4.0,180,24,1.0\n"
        )
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sites.csv"
            path.write_text(with_greenhydric, encoding="utf-8")
            self.assertEqual(4.2, load_sites(path)["SARA"].greenhydric)

            path.write_text(without_greenhydric, encoding="utf-8")
            self.assertEqual(0.0, load_sites(path)["SARA"].greenhydric)


    def test_site_loader_allows_missing_pue_but_rejects_invalid_override(self):
        missing_pue = (
            "site,max_running_jobs,green,latitude,longitude,"
            "avg_tdp_w,avg_total_cores,perf_hs06\n"
            "SARA,2,1.0,52.0,4.0,180,24,1.0\n"
        )
        invalid_pue = (
            "site,max_running_jobs,green,latitude,longitude,"
            "avg_tdp_w,avg_total_cores,perf_hs06,pue\n"
            "SARA,2,1.0,52.0,4.0,180,24,1.0,0\n"
        )
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sites.csv"
            path.write_text(missing_pue, encoding="utf-8")
            sites = load_sites(path)
            self.assertIsNone(sites["SARA"].pue)
            path.write_text(invalid_pue, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid pue"):
                load_sites(path)


if __name__ == "__main__":
    unittest.main()
