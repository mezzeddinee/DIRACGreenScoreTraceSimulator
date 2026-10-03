from __future__ import annotations

import configparser
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_URL = "https://api.wattnet.eu/token-request/get_token"
DEFAULT_API_BASE = "https://api.wattnet.eu/v1"
DEFAULT_IMPACT_PATH = "/impacts"


class WattNetHydricImpactProvider:
    """Retrieve hourly AWARE-weighted water impact from WattNet.

    Values are hydric-impact intensities in stress-L/kWh, keyed by site and
    simulated UTC hour. The provider accepts preview values but logs them.
    """

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        token_url: str = DEFAULT_TOKEN_URL,
        api_base: str = DEFAULT_API_BASE,
        impact_path: str = DEFAULT_IMPACT_PATH,
        timeout_s: float = 30.0,
        token_max_age_h: float = 24.0,
        cache_ttl_s: float = 3600.0,
    ):
        self.email = email
        self.password = password
        self.token = token
        self._token_ts = time.time() if token else None
        self.token_url = token_url
        self.api_base = api_base.rstrip("/")
        self.impact_path = impact_path
        self.timeout_s = float(timeout_s)
        self.token_max_age_h = float(token_max_age_h)
        self.cache_ttl_s = float(cache_ttl_s)
        self.session = requests.Session()
        self.cache: Dict[Tuple[str, datetime], Tuple[float, float]] = {}

    @classmethod
    def from_config(
        cls,
        conf_path: Path,
        token: Optional[str] = None,
    ) -> "WattNetHydricImpactProvider":
        cfg = configparser.ConfigParser()
        cfg.read(conf_path)
        email = os.getenv("WATTNET_EMAIL") or cfg.get(
            "WATTNET", "EMAIL", fallback=""
        ).strip()
        password = os.getenv("WATTNET_PASSWORD") or cfg.get(
            "WATTNET", "PASSWORD", fallback=""
        ).strip()
        configured_token = token or os.getenv("WATTNET_TOKEN")
        return cls(
            email=email or None,
            password=password or None,
            token=configured_token,
            token_url=cfg.get(
                "WATTNET", "TOKEN_URL", fallback=DEFAULT_TOKEN_URL
            ).strip(),
            api_base=cfg.get(
                "WATTNET", "API_BASE", fallback=DEFAULT_API_BASE
            ).strip(),
            impact_path=cfg.get(
                "WATTNET", "IMPACT_PATH", fallback=DEFAULT_IMPACT_PATH
            ).strip(),
            timeout_s=cfg.getfloat(
                "Runtime", "HYDRIC_TIMEOUT_S", fallback=30.0
            ),
            token_max_age_h=cfg.getfloat(
                "Runtime", "TOKEN_MAX_AGE_H", fallback=24.0
            ),
            cache_ttl_s=cfg.getfloat(
                "Runtime", "CACHE_TTL", fallback=3600.0
            ),
        )

    @staticmethod
    def _hour_bucket(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        return ts.replace(minute=0, second=0, microsecond=0)

    def _cache_set(self, site_name: str, bucket: datetime, value: float) -> None:
        self.cache[(site_name, bucket)] = (value, time.time())

    def _cache_get(self, site_name: str, bucket: datetime) -> Optional[float]:
        cached = self.cache.get((site_name, bucket))
        if cached is None:
            return None
        value, cached_at = cached
        if time.time() - cached_at <= self.cache_ttl_s:
            return value
        return None

    def _latest_cached_for_site(self, site_name: str) -> Optional[float]:
        entries = [
            (bucket, value)
            for (name, bucket), (value, _) in self.cache.items()
            if name == site_name
        ]
        if not entries:
            return None
        entries.sort(key=lambda item: item[0])
        return entries[-1][1]

    def _get_token(self, force: bool = False) -> str:
        if self.token and not force and self._token_ts is not None:
            age_h = (time.time() - self._token_ts) / 3600.0
            if age_h < self.token_max_age_h:
                return self.token

        if not self.email or not self.password:
            raise RuntimeError(
                "WattNet credentials are required for hydric-impact queries; "
                "set WATTNET_EMAIL/WATTNET_PASSWORD or configure [WATTNET]."
            )

        response = requests.post(
            self.token_url,
            json={"email": self.email, "password": self.password},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("WattNet token response is not valid JSON") from exc
        if not isinstance(payload, dict) or not payload.get("access_token"):
            raise RuntimeError("WattNet token response has no access_token")
        self.token = str(payload["access_token"])
        self._token_ts = time.time()
        return self.token

    @staticmethod
    def _extract_value(payload: object) -> float:
        if not isinstance(payload, list) or not payload:
            raise ValueError("Unexpected WattNet hydric-impact response")
        item = payload[0]
        if not isinstance(item, dict) or "value" not in item:
            raise ValueError("WattNet hydric-impact response has no value")
        value = float(item["value"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("WattNet hydric-impact value is invalid")
        if not item.get("valid", True):
            logger.warning(
                "using WattNet preview hydric impact zone=%s status=%s value=%s",
                item.get("zone"),
                item.get("zone_status"),
                value,
            )
        return value

    def _request_value(
        self,
        latitude: float,
        longitude: float,
        start: str,
        end: str,
    ) -> float:
        endpoint = f"{self.api_base}/{self.impact_path.lstrip('/')}"
        params = {
            "lat": float(latitude),
            "lon": float(longitude),
            "impact_type": "water",
            "scope": "operational",
            "aggregate": True,
            "use_global": True,
            "start": start,
            "end": end,
        }
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Accept": "application/json",
        }
        response = self.session.get(
            endpoint,
            params=params,
            headers=headers,
            timeout=self.timeout_s,
        )
        if response.status_code == 401:
            headers["Authorization"] = f"Bearer {self._get_token(force=True)}"
            response = self.session.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=self.timeout_s,
            )
        response.raise_for_status()
        return self._extract_value(response.json())

    def get_hydric_impact(
        self,
        site_name: str,
        impact_ts: datetime,
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> float:
        bucket = self._hour_bucket(impact_ts)
        cached = self._cache_get(site_name, bucket)
        if cached is not None:
            return cached

        stale = self._latest_cached_for_site(site_name)
        if latitude is None or longitude is None:
            if stale is not None:
                self._cache_set(site_name, bucket, stale)
                return stale
            raise RuntimeError(
                f"No coordinates or cached hydric impact for site={site_name}"
            )

        start = bucket.isoformat().replace("+00:00", "Z")
        end = (bucket + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        try:
            value = self._request_value(latitude, longitude, start, end)
        except (requests.exceptions.RequestException, ValueError, RuntimeError) as exc:
            if stale is not None:
                logger.warning(
                    "using cached hydric impact site=%s bucket=%s error=%s",
                    site_name,
                    bucket.isoformat(),
                    exc,
                )
                self._cache_set(site_name, bucket, stale)
                return stale
            raise RuntimeError(
                f"Hydric-impact query failed for site={site_name} "
                f"bucket={bucket.isoformat()} and no cached value exists"
            ) from exc

        self._cache_set(site_name, bucket, value)
        return value

    def close(self) -> None:
        self.session.close()
