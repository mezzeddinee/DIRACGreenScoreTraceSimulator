#!/usr/bin/env python3
"""Run randomized and GreenScore DIRAC simulations for four monthly traces.

The complete weekly traces are simulated. Removing the first and last day is
deliberately left to the subsequent analysis so that both boundary days remain
available as warm-up and completion context.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


SIMULATOR_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SIMULATOR_DIR.parent
SITES_FILE = SIMULATOR_DIR / "sites_original_four_with_pue.csv"
MONTHS = ("03", "04", "05", "06")
POLICIES = (
    ("random_seed42", "0"),
    ("greenscore", "1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the March--June 2026 weekly traces with randomized and "
            "GreenScore-based placement on the original four DIRAC sites."
        )
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help=(
            "Run even when a target result directory already exists. Existing "
            "files may be replaced by the simulator; nothing is deleted."
        ),
    )
    return parser.parse_args()


def simulator_python() -> Path:
    virtualenv_python = PROJECT_DIR / ".venv" / "bin" / "python"
    return virtualenv_python if virtualenv_python.is_file() else Path(sys.executable)


def validate_inputs() -> None:
    required = [SITES_FILE, SIMULATOR_DIR / "main.py"]
    required.extend(PROJECT_DIR / f"trace_2026_{month}_01_to_07.csv" for month in MONTHS)
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Required input files are missing:\n{formatted}")


def run_simulation(month: str, policy_name: str, green: str) -> None:
    trace_file = PROJECT_DIR / f"trace_2026_{month}_01_to_07.csv"
    run_label = f"original_four_sites_2026_{month}_{policy_name}"

    environment = os.environ.copy()
    environment.update(
        {
            "SIMULATOR_SITES_FILE": str(SITES_FILE),
            "SIMULATOR_TRACE_FILE": str(trace_file),
            "SIMULATOR_GREEN": green,
            "SIMULATOR_HYDRIC": "1",
            "SIMULATOR_RANDOM_SEED": "42",
            "SIMULATOR_CI_PROVIDER": "wattnet",
            "SIMULATOR_PUE_MODE": "csv",
            "SIMULATOR_ENV_TIME_MODE": "historical_interval_average",
            "SIMULATOR_SPILLOVER_DELAY_MINUTES": "0",
            "SIMULATOR_PREFERRED_SITE_COUNT": "1",
            "SIMULATOR_RUN_LABEL": run_label,
        }
    )

    print("\n" + "=" * 72, flush=True)
    print(f"Starting {run_label}", flush=True)
    print(f"Trace: {trace_file.name}", flush=True)
    print("=" * 72, flush=True)

    subprocess.run(
        [str(simulator_python()), "main.py"],
        cwd=SIMULATOR_DIR,
        env=environment,
        check=True,
    )


def main() -> None:
    args = parse_args()
    validate_inputs()

    planned_runs = [
        (month, policy_name, green)
        for month in MONTHS
        for policy_name, green in POLICIES
    ]

    if not args.allow_existing:
        existing = []
        for month, policy_name, _ in planned_runs:
            run_label = f"original_four_sites_2026_{month}_{policy_name}"
            result_dir = (
                SIMULATOR_DIR / "hydric_impact" / "timeseries" / run_label
            )
            if result_dir.exists():
                existing.append(result_dir)
        if existing:
            formatted = "\n".join(f"  - {path}" for path in existing)
            raise SystemExit(
                "Target result directories already exist:\n"
                f"{formatted}\n"
                "Use --allow-existing only if you intentionally want to rerun them."
            )

    if not os.getenv("WATTNET_TOKEN") and not (
        os.getenv("WATTNET_EMAIL") and os.getenv("WATTNET_PASSWORD")
    ):
        print(
            "Note: no WattNet credentials were found in the environment; "
            "the simulator may use its configured credential source.",
            flush=True,
        )

    print(f"Python interpreter: {simulator_python()}", flush=True)
    print(f"Site configuration: {SITES_FILE}", flush=True)
    print(f"Planned simulations: {len(planned_runs)}", flush=True)

    completed = 0
    try:
        for month, policy_name, green in planned_runs:
            run_simulation(month, policy_name, green)
            completed += 1
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Simulation failed after {completed} completed run(s), "
            f"with exit status {error.returncode}."
        ) from error
    except KeyboardInterrupt:
        raise SystemExit(f"Interrupted after {completed} completed run(s).")

    print("\nAll eight simulations completed successfully.", flush=True)
    print(
        "Results are under SIMPLIFIEDDIRACXNEW/hydric_impact/timeseries/.",
        flush=True,
    )


if __name__ == "__main__":
    main()
