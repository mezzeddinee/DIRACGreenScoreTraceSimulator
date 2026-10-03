import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wattnet_ci_provider import WattNetCarbonIntensityProvider


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class WattNetCarbonIntensityTests(unittest.TestCase):
    def test_queries_footprint_endpoint_and_caches_hour(self):
        provider = WattNetCarbonIntensityProvider(
            token="token",
            api_base="https://wattnet.example/v1",
        )
        response = _Response(
            [{"value": 123.5, "valid": True, "zone": "NL"}]
        )

        with patch.object(provider.session, "get", return_value=response) as get:
            first = provider.get_ci(
                "SARA", datetime(2026, 6, 1, 10, 5), 52.0, 4.0
            )
            second = provider.get_ci(
                "SARA", datetime(2026, 6, 1, 10, 45), 52.0, 4.0
            )

        self.assertEqual(123.5, first)
        self.assertEqual(123.5, second)
        self.assertEqual(1, get.call_count)
        args, kwargs = get.call_args
        self.assertEqual("https://wattnet.example/v1/footprints", args[0])
        self.assertEqual("carbon", kwargs["params"]["footprint_type"])
        self.assertEqual("operational", kwargs["params"]["scope"])
        self.assertTrue(kwargs["params"]["aggregate"])
        self.assertTrue(kwargs["params"]["use_global"])
        self.assertEqual("2026-06-01T10:00:00Z", kwargs["params"]["start"])
        self.assertEqual("2026-06-01T11:00:00Z", kwargs["params"]["end"])

    def test_reuses_latest_site_value_if_later_request_fails(self):
        provider = WattNetCarbonIntensityProvider(
            token="token",
            api_base="https://wattnet.example/v1",
        )
        with patch.object(
            provider,
            "_request_value",
            side_effect=[100.0, RuntimeError("service down")],
        ):
            first = provider.get_ci(
                "SARA", datetime(2026, 6, 1, 10, 5), 52.0, 4.0
            )
            second = provider.get_ci(
                "SARA", datetime(2026, 6, 1, 11, 5), 52.0, 4.0
            )

        self.assertEqual(100.0, first)
        self.assertEqual(100.0, second)


if __name__ == "__main__":
    unittest.main()
