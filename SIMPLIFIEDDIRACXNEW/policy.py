from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Dict, List, Tuple

try:
    from .models import Job, Site
except ImportError:  # direct script-style execution fallback
    from models import Job, Site

logger = logging.getLogger(__name__)


class ReplayCarbonPolicy:
    def __init__(
        self,
        green: int = 1,
        hydric_enabled: bool = False,
        spillover_delay_minutes: int = 0,
        preferred_site_count: int = 1,
    ):
        self.green = int(green)
        self.hydric_enabled = bool(hydric_enabled)
        self.spillover_delay_minutes = int(spillover_delay_minutes)
        self.preferred_site_count = int(preferred_site_count)
        if self.spillover_delay_minutes < 0:
            raise ValueError("spillover_delay_minutes must be >= 0")
        if self.preferred_site_count < 1:
            raise ValueError("preferred_site_count must be >= 1")
        self.last_site_ranks: Dict[str, int] = {}

    def estimate_green(self, sites: Dict[str, Site]) -> Dict[str, float]:
        # Hydric-aware simulations use the hydric score; carbon-only runs use
        # the original green score. In both cases, higher is better.
        if self.hydric_enabled:
            return {name: s.greenhydric for name, s in sites.items()}
        return {name: s.green for name, s in sites.items()}

    def unmet_jobs(self, waiting_jobs: List[Job], sites: Dict[str, Site]) -> List[Job]:
        slots = {name: s.available_slots() for name, s in sites.items()}
        unmet: List[Job] = []

        for job in waiting_jobs:
            assigned = False
            for site_name, site in sites.items():
                if slots[site_name] <= 0:
                    continue
                slots[site_name] -= 1
                assigned = True
                break
            if not assigned:
                unmet.append(job)
        return unmet

    def delay_applies_to_rank(self, rank: int) -> bool:
        return (
            self.green == 1
            and self.spillover_delay_minutes > 0
            and rank > self.preferred_site_count
        )

    def site_rank(self, site_name: str) -> int:
        return self.last_site_ranks.get(site_name, 0)

    def _eligible_for_spillover(self, jobs: List[Job], current_time: datetime) -> int:
        """Count the FIFO prefix old enough to use a lower-ranked site."""
        eligible = 0
        for job in jobs:
            waited_minutes = (current_time - job.submit_time).total_seconds() / 60.0
            if waited_minutes < self.spillover_delay_minutes:
                break
            eligible += 1
        return eligible

    def schedule(
        self,
        waiting_jobs: List[Job],
        sites: Dict[str, Site],
        current_time: datetime | None = None,
    ) -> List[Tuple[str, int]]:
        # Preserve the caller-defined waiting order (sorted in simulator step_match).
        remaining = list(waiting_jobs)
        demand = len(remaining)
        if demand <= 0:
            logger.debug("schedule no demand")
            return []

        if self.green == 1:
            green_score = self.estimate_green(sites)
            # Score mode: rank by greenhydric when hydric accounting is enabled,
            # otherwise rank by the original green score (higher is better).
            scored = sorted(sites.values(), key=lambda s: green_score[s.name], reverse=True)
        else:
            # Non-green mode: random site order.
            scored = list(sites.values())
            random.shuffle(scored)

        self.last_site_ranks = {
            site.name: rank for rank, site in enumerate(scored, start=1)
        }

        delay_enabled = self.green == 1 and self.spillover_delay_minutes > 0
        if delay_enabled and current_time is None:
            raise ValueError("current_time is required when spillover delay is enabled")

        submissions: List[Tuple[str, int]] = []
        for rank, site in enumerate(scored, start=1):
            avail = site.available_slots()
            if avail <= 0:
                continue

            eligible_demand = demand
            if self.delay_applies_to_rank(rank):
                assert current_time is not None
                eligible_demand = self._eligible_for_spillover(remaining, current_time)
            x = min(eligible_demand, avail)
            if x <= 0:
                continue

            submissions.append((site.name, x))
            del remaining[:x]
            demand -= x
            if demand == 0:
                break

        logger.info(
            "schedule green=%d demand=%d spillover_delay_min=%d preferred_sites=%d submissions=%s",
            self.green,
            len(waiting_jobs),
            self.spillover_delay_minutes if self.green == 1 else 0,
            self.preferred_site_count,
            submissions,
        )
        return submissions
