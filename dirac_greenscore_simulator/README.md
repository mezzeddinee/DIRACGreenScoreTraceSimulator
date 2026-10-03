# DIRAC GreenScore Trace Simulator

Independent simplified-DIRAC variant for bounded-delay experiments:
- No pilot objects.
- Jobs are assigned directly to sites.
- Each site has a max number of concurrent running jobs (`max_running_jobs`).
- Score-based site ranking uses `greenhydric` during hydric simulations and
  falls back to `greenscore` (`green`) when hydric simulation is disabled.
- No tag-based compatibility filtering: any waiting job can run on any site.
- In green mode, lower-ranked sites can be gated until a FIFO job has waited a
  configurable number of minutes.
- Carbon and hydric intensities default to averaging over the same fixed
  historical trace interval for every scheduling policy.

## Input Files

- `sites.csv`: site capacity and characteristics. `greenhydric` defaults to `0` when absent. PUE is resolved from CIM `/pue` by `site_name`; an optional positive `pue` column is used only as an offline fallback. `max_running_jobs` falls back to `max_pilots` when absent.
- `jobs.csv` / `trace.csv`:
  - base fields: `job_id,submit_time,norm_cpu_seconds,cores_used`
  - runtime-related fields: `wallclock` and `CPUNormFactor`
  - supported header aliases from DIRAC traces:
    - `wallclock` / `wallclocktime` / `WallClockTime` / `WallClockTime(s)`
    - `CPUNormFactor` / `CPUNormalizationFactor` / `cpunormlazationfactor`
    - `norm_cpu_seconds` / `cpu_seconds` / `NormCPUTime(s)`
  - if only `runtime_min` is present, wallclock seconds are derived as `runtime_min * 60`
- `SIMULATOR_TRACE_FILE`: trace CSV path, resolved relative to the simulator
  directory unless absolute; defaults to `trace_2026_06_01.csv`.
- `cim.conf`: optional CIM/KPI configuration plus WattNet carbon and hydric-impact configuration. WattNet credentials can be supplied with `WATTNET_EMAIL`/`WATTNET_PASSWORD` or `WATTNET_TOKEN`.

## Step Flow (1 minute per tick)

1. `release_jobs()`
- Move jobs from `pending` to `waiting` when `submit_time <= current_time`.

2. `step_match()`
- Policy computes unmet waiting demand and returns `(site, k)` submissions.
- Simulator starts up to `k` waiting jobs on each site, limited by `available_slots()`.
- For each started job:
  - state becomes `running`
  - site-specific runtime is derived from job parameters and site performance:
    - `cpu_seconds_sim = norm_cpu_seconds / perf_hs06`
    - `wallclock_seconds_sim = (wallclock * cpu_norm_factor) / perf_hs06`
    - `runtime_min = ceil(wallclock_seconds_sim / 60)`
  - total energy is computed
  - carbon intensity is averaged over the reconstructed historical trace
    interval directly from WattNet's operational carbon-footprint endpoint
  - carbon is computed once: `carbon_kg = total_energy_kwh * ci / 1000`
  - site PUE is read from `sites.csv` by default; CIM/KPI `/pue` remains optional
  - WattNet hydric-impact intensity is averaged over the same trace interval
  - facility-aware water impact is computed once: `stress_L = total_energy_kwh * site_pue * HI_stress_L_per_kWh`

3. `step_execute()`
- Decrement `remaining_min` for each running job.
- Completed jobs move to `done` with `finish_time` set.

4. Advance time by one tick.

## Policy

- Unmet demand is computed against currently free site slots.
- `green` flag controls site ordering:
  - `green=1`: rank descending by `greenhydric` when hydric simulation is enabled, otherwise by `green` (higher is better).
  - `green=0`: ignore `green` and shuffle sites randomly before assignment.
- Runtime switch is done with `SIMULATOR_GREEN` (default is `0`).

### Bounded spillover delay

In score-based mode, the highest-ranked site is preferred by default. Jobs
that cannot start there remain in the FIFO queue until either preferred-site
capacity becomes available or their queue age reaches the configured delay.
Once the threshold is reached, lower-ranked sites may accept them.

- `SIMULATOR_SPILLOVER_DELAY_MINUTES`: non-negative delay threshold; default `0`.
- `SIMULATOR_PREFERRED_SITE_COUNT`: number of top-ranked sites exempt from the
  gate; default `1`.
- `SIMULATOR_ENV_TIME_MODE`: `historical_interval_average` (default),
  `execution_midpoint`, `execution_start`, or `submission`. The historical
  interval runs from original submission time through submission plus the
  historical wall-clock duration and is identical across policies.
- `SIMULATOR_CI_TIMESTAMP_MODE`: deprecated compatibility alias for
  `SIMULATOR_ENV_TIME_MODE`; its selected point mode applies to both signals.
- `SIMULATOR_CI_PROVIDER`: `wattnet` (default) or `cim`.
- `SIMULATOR_PUE_MODE`: `api_with_fallback` (default), `api`, or `csv`.
  Use `csv` to skip the CIM/KPI PUE endpoint and use the audited values in
  `sites.csv`. When WattNet supplies CI, the default PUE mode is `csv`, so the
  run makes no CIM requests.

The delay gate applies only when `SIMULATOR_GREEN=1`. Random scheduling remains
an eager baseline. See `DELAY_POLICY.md` for the precise algorithm and experiment
matrix.

## Run

```bash
cd /path/to/DIRACGreenScoreTraceSimulator
python3 -m venv .venv
.venv/bin/python -m pip install -r dirac_greenscore_simulator/requirements.txt
cd dirac_greenscore_simulator
# Green mode (score-based ranking) with hydric-impact accounting
WATTNET_EMAIL=... WATTNET_PASSWORD=... SIMULATOR_GREEN=1 python3 main.py
# Green mode with a 60-minute gate before spillover beyond the best site
WATTNET_EMAIL=... WATTNET_PASSWORD=... \
SIMULATOR_GREEN=1 SIMULATOR_HYDRIC=1 \
SIMULATOR_SPILLOVER_DELAY_MINUTES=60 \
SIMULATOR_PREFERRED_SITE_COUNT=1 \
SIMULATOR_PUE_MODE=csv \
SIMULATOR_RUN_LABEL=green_delay60 \
python3 main.py
# Non-green mode (randomized site ordering) with hydric-impact accounting
WATTNET_EMAIL=... WATTNET_PASSWORD=... SIMULATOR_GREEN=0 python3 main.py
# Legacy carbon-only mode
SIMULATOR_HYDRIC=0 SIMULATOR_GREEN=0 python3 main.py
```
