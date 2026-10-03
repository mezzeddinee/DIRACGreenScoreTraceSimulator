from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path

try:
    from .ci_provider import MidpointCIProvider
    from .csv_io import load_jobs, load_sites
    from .hydric_impact_provider import WattNetHydricImpactProvider
    from .metrics import print_summary
    from .policy import ReplayCarbonPolicy
    from .report_plots import save_report_plots
    from .simulator import ReplaySimulator
    from wattnet_ci_provider import WattNetCarbonIntensityProvider
except ImportError:  # direct script-style execution fallback
    from ci_provider import MidpointCIProvider
    from csv_io import load_jobs, load_sites
    from hydric_impact_provider import WattNetHydricImpactProvider
    from metrics import print_summary
    from policy import ReplayCarbonPolicy
    from report_plots import save_report_plots
    from simulator import ReplaySimulator
    from wattnet_ci_provider import WattNetCarbonIntensityProvider

logger = logging.getLogger(__name__)


def run(base: Path, tick_minutes: int = 1) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    logger.info("run start base=%s tick=%d", base, tick_minutes)
    sites_file = Path(os.getenv("SIMULATOR_SITES_FILE", "sites.csv"))
    sites_path = sites_file if sites_file.is_absolute() else base / sites_file
    sites = load_sites(sites_path)
    trace_file = Path(
        os.getenv("SIMULATOR_TRACE_FILE", "trace_2026_06_01.csv")
    )
    trace_path = trace_file if trace_file.is_absolute() else base / trace_file
    jobs = load_jobs(trace_path)
    logger.info(
        "input loaded sites=%d jobs=%d sites_path=%s trace_path=%s",
        len(sites),
        len(jobs),
        sites_path,
        trace_path,
    )

    conf_path = base / "cim.conf"
    token = None
    cim_provider = MidpointCIProvider.from_config(
        conf_path=conf_path,
        token=token,
    )
    ci_provider_mode = os.getenv("SIMULATOR_CI_PROVIDER", "wattnet").strip().lower()
    if ci_provider_mode == "wattnet":
        ci_provider = WattNetCarbonIntensityProvider.from_config(conf_path=conf_path)
    elif ci_provider_mode == "cim":
        ci_provider = cim_provider
    else:
        raise ValueError("SIMULATOR_CI_PROVIDER must be wattnet or cim")
    logger.info("ci provider configured provider=%s conf=%s", ci_provider_mode, conf_path)
    hydric_enabled = int(os.getenv("SIMULATOR_HYDRIC", "1")) == 1
    hydric_impact_provider = (
        WattNetHydricImpactProvider.from_config(conf_path=conf_path)
        if hydric_enabled
        else None
    )
    logger.info("hydric impact configured enabled=%s", hydric_enabled)
    default_pue_mode = "csv" if ci_provider_mode == "wattnet" else "api_with_fallback"
    pue_mode = os.getenv("SIMULATOR_PUE_MODE", default_pue_mode).strip().lower()
    if pue_mode not in {"api", "api_with_fallback", "csv"}:
        raise ValueError(
            "SIMULATOR_PUE_MODE must be api, api_with_fallback, or csv"
        )
    if hydric_enabled:
        for site in sites.values():
            configured_pue = site.pue
            if pue_mode == "csv":
                if configured_pue is None or float(configured_pue) <= 0.0:
                    raise ValueError(
                        f"Valid CSV PUE required in csv mode for site={site.name}"
                    )
                logger.info(
                    "using configured CSV PUE site=%s pue=%.4f",
                    site.name,
                    configured_pue,
                )
                continue
            logger.info("resolving PUE through API site=%s", site.name)
            try:
                site.pue = cim_provider.get_pue(
                    site_name=site.name,
                    latitude=site.latitude,
                    longitude=site.longitude,
                )
            except RuntimeError:
                if pue_mode == "api" or configured_pue is None:
                    raise
                site.pue = configured_pue
                logger.warning(
                    "using configured PUE fallback site=%s pue=%.4f",
                    site.name,
                    site.pue,
                )
    green = int(os.getenv("SIMULATOR_GREEN", "0"))
    spillover_delay_minutes = int(
        os.getenv("SIMULATOR_SPILLOVER_DELAY_MINUTES", "0")
    )
    preferred_site_count = int(os.getenv("SIMULATOR_PREFERRED_SITE_COUNT", "1"))
    legacy_time_mode = os.getenv("SIMULATOR_CI_TIMESTAMP_MODE", "").strip()
    environmental_time_mode = os.getenv(
        "SIMULATOR_ENV_TIME_MODE",
        legacy_time_mode or "historical_interval_average",
    ).strip()
    if legacy_time_mode and not os.getenv("SIMULATOR_ENV_TIME_MODE"):
        logger.warning(
            "SIMULATOR_CI_TIMESTAMP_MODE is deprecated; applying it to both "
            "carbon and water. Use SIMULATOR_ENV_TIME_MODE instead."
        )
    random_seed_raw = os.getenv("SIMULATOR_RANDOM_SEED", "").strip()
    if random_seed_raw:
        random_seed = int(random_seed_raw)
        random.seed(random_seed)
        logger.info("random seed configured seed=%d", random_seed)
    policy = ReplayCarbonPolicy(
        green=green,
        hydric_enabled=hydric_enabled,
        spillover_delay_minutes=spillover_delay_minutes,
        preferred_site_count=preferred_site_count,
    )
    ranking_score = "random" if green != 1 else ("greenhydric" if hydric_enabled else "green")
    logger.info(
        "policy configured green=%d ranking_score=%s spillover_delay_min=%d preferred_sites=%d",
        green,
        ranking_score,
        spillover_delay_minutes if green == 1 else 0,
        preferred_site_count,
    )

    sim = ReplaySimulator(
        sites=sites,
        jobs=jobs,
        tick_minutes=tick_minutes,
        policy=policy,
        ci_provider=ci_provider,
        hydric_impact_provider=hydric_impact_provider,
        environmental_time_mode=environmental_time_mode,
    )

    steps = 0
    heartbeat_every = 1000
    while not sim.done():
        sim.step()
        steps += 1
        if steps % heartbeat_every == 0:
            waiting = len(sim.waiting_jobs())
            active = sim.active_jobs()
            done = len(sim.done_jobs)
            logger.info(
                "heartbeat steps=%d t=%s waiting=%d active=%d done=%d",
                steps,
                sim.current_time.isoformat(),
                waiting,
                active,
                done,
            )

    logger.info("run done steps=%d done_jobs=%d", steps, len(sim.done_jobs))
    print_summary(sim.done_jobs)
    run_label = os.getenv("SIMULATOR_RUN_LABEL", "").strip()
    if run_label:
        if re.fullmatch(r"[A-Za-z0-9_.-]+", run_label) is None:
            raise ValueError(
                "SIMULATOR_RUN_LABEL may contain only letters, numbers, '.', '_' and '-'"
            )
        report_dir = base / "hydric_impact" / "timeseries" / run_label
    else:
        report_dir = base / "plots"
    save_report_plots(sim, out_dir=report_dir)
    if hydric_impact_provider is not None:
        hydric_impact_provider.close()
    if ci_provider is not cim_provider:
        ci_provider.close()
