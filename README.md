# Simplified DIRAC Trace-Replay Simulator

This repository contains a trace-driven simulator used to compare randomized
and GreenScore-based scheduling across DIRAC sites. It includes four seven-day
job traces from March through June 2026, the four-site configuration used by
the controlled comparison, and a launcher that reproduces the complete
eight-run simulation matrix.

The simulator advances in one-minute ticks. Jobs are released in submission
order, assigned directly to sites subject to site capacity, and executed using
site-specific performance and energy characteristics. In GreenScore mode,
sites are ranked by the static `greenhydric` values in the selected site CSV.

## Repository Layout

- `SIMPLIFIEDDIRACXNEW/`: simulator source, tests, configuration, and plotting
  utilities.
- `SIMPLIFIEDDIRACXNEW/sites_original_four_with_pue.csv`: controlled four-site
  configuration containing IN2P3-IRES, SARA-MATRIX, FZK-LCG2, and RAL-LCG2.
- `SIMPLIFIEDDIRACXNEW/run_original_four_sites_four_traces.py`: launcher for
  the complete March--June randomized/GreenScore matrix.
- `trace_2026_03_01_to_07.csv`: March trace, 727,014 jobs.
- `trace_2026_04_01_to_07.csv`: April trace, 700,514 jobs.
- `trace_2026_05_01_to_07.csv`: May trace, 341,275 jobs.
- `trace_2026_06_01_to_07.csv`: June trace, 997,370 jobs.
- `trace_2026_06_01_to_03.csv`: shorter three-day trace.
- `SIMPLIFIEDDIRACXNEW/trace_2026_06_01.csv`: default one-day trace.

Generated simulation results are written below
`SIMPLIFIEDDIRACXNEW/hydric_impact/timeseries/` and are intentionally excluded
from Git.

## Requirements

- Python 3.10 or newer; the current project was verified with Python 3.12.
- Network access to the WattNet API.
- A WattNet token, or WattNet email/password credentials.
- `requests`, installed from the supplied requirements file.
- `matplotlib` if PNG report plots are required. Without it, the simulator
  still exports the CSV results.

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r SIMPLIFIEDDIRACXNEW/requirements.txt
.venv/bin/python -m pip install matplotlib
```

Run the unit tests before a long simulation:

```bash
cd SIMPLIFIEDDIRACXNEW
../.venv/bin/python -m unittest discover -s tests -v
cd ..
```

The current suite contains 37 tests.

## Authentication

Do not place live credentials in Git. The tracked
`SIMPLIFIEDDIRACXNEW/cim.conf.example` contains only non-secret settings. The
recommended approach is to provide credentials through environment variables.

Using a token:

```bash
read -rsp "WattNet token: " WATTNET_TOKEN
export WATTNET_TOKEN
```

Or using email and password:

```bash
read -rp "WattNet email: " WATTNET_EMAIL
read -rsp "WattNet password: " WATTNET_PASSWORD
export WATTNET_EMAIL WATTNET_PASSWORD
```

A local `SIMPLIFIEDDIRACXNEW/cim.conf` may also be created from the example,
but that file is ignored by Git because it may contain credentials:

```bash
cp SIMPLIFIEDDIRACXNEW/cim.conf.example SIMPLIFIEDDIRACXNEW/cim.conf
```

## Reproduce the Four-Trace Simulation Matrix

The standard experiment runs each March, April, May, and June trace twice:

1. randomized scheduling with seed 42;
2. GreenScore-based scheduling using the static `greenhydric` site order.

The launcher fixes the following settings for all eight runs:

- sites: `sites_original_four_with_pue.csv`;
- hydric accounting enabled;
- WattNet carbon intensity and water-scarcity intensity;
- audited PUE values from the site CSV;
- `historical_interval_average` environmental timing;
- no spillover delay;
- one preferred site;
- random seed 42.

After exporting WattNet credentials, run from the repository root:

```bash
.venv/bin/python SIMPLIFIEDDIRACXNEW/run_original_four_sites_four_traces.py
```

The expected output directories are:

```text
SIMPLIFIEDDIRACXNEW/hydric_impact/timeseries/
├── original_four_sites_2026_03_random_seed42/
├── original_four_sites_2026_03_greenscore/
├── original_four_sites_2026_04_random_seed42/
├── original_four_sites_2026_04_greenscore/
├── original_four_sites_2026_05_random_seed42/
├── original_four_sites_2026_05_greenscore/
├── original_four_sites_2026_06_random_seed42/
└── original_four_sites_2026_06_greenscore/
```

The launcher refuses to start if any target directory already exists. This
prevents accidental replacement of previous results. A fresh clone or a new
workspace is recommended for a clean reproduction. To deliberately rerun into
existing directories, use:

```bash
.venv/bin/python SIMPLIFIEDDIRACXNEW/run_original_four_sites_four_traces.py --allow-existing
```

This option may replace files but does not clean the directories first.

The launcher simulates all seven days of every trace. The scientific analysis
used days 1 and 7 as boundary context and retained only jobs submitted during
days 2--6. That boundary trimming is a downstream analysis operation; it is
not performed by the simulator or the eight-run launcher.

## Run One Trace Manually

Run commands from `SIMPLIFIEDDIRACXNEW/`. Always choose a new run label unless
you intentionally want to write into an existing result directory.

### GreenScore-based scheduling

```bash
cd SIMPLIFIEDDIRACXNEW

SIMULATOR_SITES_FILE=sites_original_four_with_pue.csv \
SIMULATOR_TRACE_FILE=../trace_2026_06_01_to_07.csv \
SIMULATOR_GREEN=1 \
SIMULATOR_HYDRIC=1 \
SIMULATOR_CI_PROVIDER=wattnet \
SIMULATOR_PUE_MODE=csv \
SIMULATOR_ENV_TIME_MODE=historical_interval_average \
SIMULATOR_SPILLOVER_DELAY_MINUTES=0 \
SIMULATOR_PREFERRED_SITE_COUNT=1 \
SIMULATOR_RUN_LABEL=june_greenscore \
../.venv/bin/python main.py
```

### Randomized baseline

```bash
SIMULATOR_SITES_FILE=sites_original_four_with_pue.csv \
SIMULATOR_TRACE_FILE=../trace_2026_06_01_to_07.csv \
SIMULATOR_GREEN=0 \
SIMULATOR_HYDRIC=1 \
SIMULATOR_RANDOM_SEED=42 \
SIMULATOR_CI_PROVIDER=wattnet \
SIMULATOR_PUE_MODE=csv \
SIMULATOR_ENV_TIME_MODE=historical_interval_average \
SIMULATOR_SPILLOVER_DELAY_MINUTES=0 \
SIMULATOR_RUN_LABEL=june_random_seed42 \
../.venv/bin/python main.py
```

## Run Any Compatible Trace

Set `SIMULATOR_TRACE_FILE` to an absolute path or a path relative to
`SIMPLIFIEDDIRACXNEW/`. A compatible CSV must contain:

```text
job_id,submit_time,norm_cpu_seconds,cores_used
```

It must also provide either `wallclock`/`wallclocktime` and a CPU normalization
factor, or `runtime_min`. Supported aliases are documented in
`SIMPLIFIEDDIRACXNEW/README.md`.

Example with a custom trace and the default site configuration:

```bash
cd SIMPLIFIEDDIRACXNEW

SIMULATOR_TRACE_FILE=/absolute/path/to/custom_trace.csv \
SIMULATOR_SITES_FILE=sites.csv \
SIMULATOR_GREEN=1 \
SIMULATOR_HYDRIC=1 \
SIMULATOR_CI_PROVIDER=wattnet \
SIMULATOR_PUE_MODE=csv \
SIMULATOR_ENV_TIME_MODE=historical_interval_average \
SIMULATOR_RUN_LABEL=custom_trace_greenscore \
../.venv/bin/python main.py
```

To test bounded spillover, add for example:

```bash
SIMULATOR_SPILLOVER_DELAY_MINUTES=60 \
SIMULATOR_PREFERRED_SITE_COUNT=1
```

The delay gate applies only to GreenScore mode. Randomized scheduling remains
an eager baseline.

## Outputs

Each labelled result directory contains:

- `completed_jobs.csv`: job-level placement, timing, energy, carbon, water,
  environmental interval, and spillover information;
- `waiting_over_time.csv`;
- `jobs_submitted_per_site_over_time.csv`;
- `jobs_running_per_site_over_time.csv`;
- PNG workload and queue plots when `matplotlib` is installed.

The console summary reports completed jobs, queue and turnaround percentiles,
carbon, normalized CPU time, water-scarcity impact, and site allocations.

For a basic completion check, the number of data rows in `completed_jobs.csv`
must equal the number of data rows in the corresponding input trace.

## Important Accounting Details

- GreenScore scheduling currently uses static `greenhydric` values from the
  selected site CSV; it does not recompute a time-varying Green Score during
  scheduling.
- Historical environmental intervals are reconstructed as job submission time
  plus historical wall-clock duration because the traces do not contain the
  original execution-start timestamp.
- `water_impact_stress_l` is calculated as IT energy times PUE times WattNet
  water-scarcity intensity. Its unit is AWARE-weighted stress-L, not physical
  water volume.
- The raw simulator `carbon_kg` field is calculated as IT energy times carbon
  intensity and does **not** include PUE. The poster analysis recomputed
  facility-adjusted carbon as IT energy times PUE times carbon intensity. Use
  the same post-processing convention when comparing against published poster
  values.
- The scientific production comparisons and the controlled trace replays have
  different evidential meanings. Only the paired trace replays isolate the
  scheduling policy under common workload and infrastructure inputs.

For simulator internals and all environment variables, see
`SIMPLIFIEDDIRACXNEW/README.md` and `SIMPLIFIEDDIRACXNEW/DELAY_POLICY.md`.
