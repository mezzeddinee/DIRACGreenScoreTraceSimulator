from __future__ import annotations

import configparser
import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class MidpointCIProvider:
    def __init__(
        self,
        token: Optional[str] = None,
        kpi_api_base: str = "",
        cim_api_base: str = "",
        email: Optional[str] = None,
        password: Optional[str] = None,
        default_pue: float = 1.4,
        fallback_ci: float = 300.0,
        timeout_s: float = 5.0,
        pue_timeout_s: float = 20.0,
        token_max_age_h: float = 24.0,
        ci_cache_ttl_s: float = 3600.0,
    ):
        self.token = token
        self._token_ts: Optional[float] = None
        self.base = kpi_api_base.rstrip("/")
        self.cim_api_base = cim_api_base.rstrip("/")
        self.email = email
        self.password = password
        self.default_pue = default_pue
        self.fallback_ci = fallback_ci
        self.timeout_s = timeout_s
        self.pue_timeout_s = pue_timeout_s
        self.token_max_age_h = token_max_age_h
        self.ci_cache_ttl_s = ci_cache_ttl_s
        # Cache layout:
        #   key   = (site_name, bucket_datetime)
        #   value = (ci_gco2_per_kwh, cached_unix_timestamp)
        self.cache: Dict[Tuple[str, datetime], Tuple[float, float]] = {}
        self.pue_cache: Dict[str, float] = {}

    def _cache_set(self, site_name: str, bucket: datetime, ci: float) -> None:
        # Store one CI value per site/time bucket; stale entries are ignored by _cache_get.
        self.cache[(site_name, bucket)] = (ci, time.time())
        logger.info("ci cache set site=%s bucket=%s ci=%.3f", site_name, bucket.isoformat(), ci)

    def _cache_get(self, site_name: str, bucket: datetime) -> Optional[float]:
        key = (site_name, bucket)
        cached = self.cache.get(key)
        if cached is None:
            return None

        ci, cached_ts = cached
        age_s = time.time() - cached_ts
        if age_s <= self.ci_cache_ttl_s:
            return ci

        logger.info(
            "ci cache expired site=%s bucket=%s age_s=%.1f ttl_s=%.1f",
            site_name,
            bucket.isoformat(),
            age_s,
            self.ci_cache_ttl_s,
        )
        return None

    def _latest_cached_ci_for_site(self, site_name: str) -> Optional[float]:
        site_entries = [(bucket, ci) for (name, bucket), (ci, _) in self.cache.items() if name == site_name]
        if not site_entries:
            return None
        site_entries.sort(key=lambda x: x[0])
        return site_entries[-1][1]

    @classmethod
    def from_config(
        cls,
        conf_path: Path,
        token: Optional[str] = None,
    ) -> "MidpointCIProvider":
        cfg = configparser.ConfigParser()
        cfg.read(conf_path)
        cim_api_base = cfg.get("CIM", "API_BASE", fallback="").strip()
        email = cfg.get("CIM", "EMAIL", fallback="").strip() or None
        password = cfg.get("CIM", "PASSWORD", fallback="").strip() or None
        kpi_api_base = cfg.get("KPI", "API_BASE", fallback="").strip()
        default_pue = cfg.getfloat("Defaults", "PUE", fallback=1.4)
        fallback_ci = cfg.getfloat("Defaults", "CI", fallback=300.0)
        ci_timeout_s = cfg.getfloat("Runtime", "CI_TIMEOUT_S", fallback=5.0)
        pue_timeout_s = cfg.getfloat("Runtime", "PUE_TIMEOUT_S", fallback=20.0)
        token_max_age_h = cfg.getfloat("Runtime", "TOKEN_MAX_AGE_H", fallback=24.0)
        ci_cache_ttl_s = cfg.getfloat("Runtime", "CACHE_TTL", fallback=3600.0)
        return cls(
            token=token,
            kpi_api_base=kpi_api_base,
            cim_api_base=cim_api_base,
            email=email,
            password=password,
            default_pue=default_pue,
            fallback_ci=fallback_ci,
            timeout_s=ci_timeout_s,
            pue_timeout_s=pue_timeout_s,
            token_max_age_h=token_max_age_h,
            ci_cache_ttl_s=ci_cache_ttl_s,
        )

    def _get_token(self) -> Optional[str]:
        if self.token and self._token_ts is not None:
            age_h = (time.time() - self._token_ts) / 3600.0
            if age_h < self.token_max_age_h:
                logger.info("token cache hit age_h=%.2f", age_h)
                return self.token

        if self.token and self._token_ts is None:
            self._token_ts = time.time()
            logger.info("token provided externally")
            return self.token

        if not self.cim_api_base or not self.email or not self.password:
            logger.info("token fetch skipped missing cim credentials")
            return None

        try:
            resp = requests.get(
                f"{self.cim_api_base}/token",
                params={"email": self.email, "password": self.password},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            token = (
                data.get("access_token")
                or data.get("token")
                or data.get("jwt")
                or data.get("id_token")
            )
            if token:
                self.token = token
                self._token_ts = time.time()
                logger.info("token fetched")
                return token
            logger.info("token fetch empty response")
            return None
        except (
            requests.exceptions.RequestException,
            ValueError,
            json.JSONDecodeError,
        ):
            logger.info("token fetch failed; using anonymous mode")
            return None

    def _hour_bucket(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        return ts.replace(minute=0, second=0, microsecond=0)

    def get_pue(
        self,
        site_name: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> float:
        """Resolve a site's PUE through the CIM/KPI /pue endpoint."""
        if site_name in self.pue_cache:
            return self.pue_cache[site_name]

        token = self._get_token()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = requests.post(
                f"{self.base}/pue",
                json={"site_name": site_name},
                headers=headers,
                timeout=self.pue_timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
            pue = float(payload["pue"])
            if not math.isfinite(pue) or pue <= 0.0:
                raise ValueError(f"invalid pue={pue}")

            location = payload.get("location") or {}
            cim_lat = location.get("latitude")
            cim_lon = location.get("longitude")
            if (
                latitude is not None
                and longitude is not None
                and cim_lat is not None
                and cim_lon is not None
                and (
                    abs(float(latitude) - float(cim_lat)) > 1.0
                    or abs(float(longitude) - float(cim_lon)) > 1.0
                )
            ):
                logger.warning(
                    "CIM location differs from CSV site=%s csv=(%s,%s) cim=(%s,%s)",
                    site_name,
                    latitude,
                    longitude,
                    cim_lat,
                    cim_lon,
                )
        except (
            requests.exceptions.RequestException,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(f"CIM PUE lookup failed for site={site_name}") from exc

        self.pue_cache[site_name] = pue
        logger.info("pue api ok site=%s pue=%.4f", site_name, pue)
        return pue

    def get_ci(
        self,
        site_name: str,
        midpoint_ts: datetime,
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> float:
        bucket = self._hour_bucket(midpoint_ts)
        ci_cached = self._cache_get(site_name, bucket)
        if ci_cached is not None:
            logger.debug("ci cache hit site=%s bucket=%s ci=%.3f", site_name, bucket.isoformat(), ci_cached)
            return ci_cached
        ci_stale = self._latest_cached_ci_for_site(site_name)

        if latitude is None or longitude is None:
            if ci_stale is not None:
                self._cache_set(site_name, bucket, ci_stale)
                logger.info(
                    "ci cache reuse site=%s bucket=%s ci=%.3f reason=missing_coords",
                    site_name,
                    bucket.isoformat(),
                    ci_stale,
                )
                return ci_stale
            raise RuntimeError(
                f"No cached CI available for site={site_name}; cannot compute CI without coordinates."
            )

        start = bucket.isoformat().replace("+00:00", "Z")
        end = (bucket + timedelta(hours=1)).isoformat().replace("+00:00", "Z")

        payload = {
            "lat": latitude,
            "lon": longitude,
            "pue": self.default_pue,
            "energy_wh": 1000,
            "start": start,
            "end": end,
            "metric_id": f"{site_name}_{bucket.isoformat()}",
            "wattnet_params": {"granularity": "hour"},
        }
        token = self._get_token()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = requests.post(
                f"{self.base}/ci",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            if "ci_gco2_per_kwh" not in data:
                raise ValueError("ci_gco2_per_kwh missing in KPI response")
            ci = float(data["ci_gco2_per_kwh"])
            logger.info("ci api ok site=%s bucket=%s ci=%.3f", site_name, bucket.isoformat(), ci)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "n/a"
            body = (exc.response.text[:300] if exc.response is not None and exc.response.text else "").replace("\n", " ")
            if ci_stale is not None:
                self._cache_set(site_name, bucket, ci_stale)
                logger.info(
                    "ci api http_error site=%s bucket=%s status=%s body=%s using_cached ci=%.3f",
                    site_name,
                    bucket.isoformat(),
                    status,
                    body,
                    ci_stale,
                )
                return ci_stale
            raise RuntimeError(
                f"CI request failed for site={site_name} bucket={bucket.isoformat()} status={status} and no cached CI exists."
            ) from exc
        except requests.exceptions.RequestException as exc:
            if ci_stale is not None:
                self._cache_set(site_name, bucket, ci_stale)
                logger.info(
                    "ci api request_error site=%s bucket=%s err=%s using_cached ci=%.3f",
                    site_name,
                    bucket.isoformat(),
                    exc,
                    ci_stale,
                )
                return ci_stale
            raise RuntimeError(
                f"CI request error for site={site_name} bucket={bucket.isoformat()} and no cached CI exists."
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            if ci_stale is not None:
                self._cache_set(site_name, bucket, ci_stale)
                logger.info(
                    "ci api parse_error site=%s bucket=%s err=%s using_cached ci=%.3f",
                    site_name,
                    bucket.isoformat(),
                    exc,
                    ci_stale,
                )
                return ci_stale
            raise RuntimeError(
                f"CI response parse error for site={site_name} bucket={bucket.isoformat()} and no cached CI exists."
            ) from exc

        self._cache_set(site_name, bucket, ci)
        return ci
