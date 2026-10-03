# Bounded spillover policy

## Objective

The policy tests whether a controlled increase in queueing delay can avoid
placing jobs at lower-ranked sites. It is intended to produce an environmental
impact versus latency trade-off rather than to minimize impact without limits.

## Decision rule

At every one-minute scheduling tick:

1. Waiting jobs remain in submission-time FIFO order.
2. Sites are ranked by decreasing `greenhydric` in hydric mode, or by `green`
   in carbon-only mode.
3. The first `SIMULATOR_PREFERRED_SITE_COUNT` sites accept waiting jobs
   immediately when they have free capacity.
4. All lower-ranked sites are closed to a job until its queue age reaches
   `SIMULATOR_SPILLOVER_DELAY_MINUTES`.
5. At the exact threshold the job becomes eligible for lower-ranked sites.
6. FIFO is preserved: the scheduler never skips a younger ineligible job to
   start a later job.

With a threshold of zero, behavior is identical to eager score-based
scheduling. The gate is disabled in random mode so that random remains an eager
baseline.

## Environmental timestamps

The default mode is `historical_interval_average`. Both carbon and water are
averaged over a reconstructed trace interval beginning at the original job
submission time and lasting for the historical wall-clock duration. The trace
does not contain the historical execution-start timestamp, so this interval is
an explicit approximation. It is fixed across scheduling policies and isolates
spatial placement from simulated queueing time.

Set `SIMULATOR_ENV_TIME_MODE=execution_midpoint`, `execution_start`, or
`submission` for alternative experiments. A selected point mode applies to
both environmental signals.

Carbon intensity is obtained directly from WattNet by default. Set
`SIMULATOR_CI_PROVIDER=cim` only for compatibility comparisons with earlier
runs. With the default WattNet CI and CSV PUE modes, CIM is not contacted.

The supplied `sites.csv` also contains the PUE values established in the
previous experiments. Set `SIMULATOR_PUE_MODE=csv` to use them directly when
the CIM/KPI PUE service is unavailable.

## Auditing

`completed_jobs.csv` adds:

- `queue_delay_min`
- `site_rank_at_start`
- `spillover_delay_threshold_min`

These fields allow the delayed jobs, realized queue cost, and lower-ranked-site
placements to be measured directly.

## Recommended experiment matrix

Use the same trace, sites, environmental inputs and random seed for all runs:

- eager green: delay 0 minutes;
- delayed green: delays 30, 60, 120 and 240 minutes;
- eager random baseline: seed 42;
- optionally repeat the random baseline with at least 30 seeds.

Report total carbon, water-scarcity impact, combined EI burden, p50/p95/p99
queue delay, makespan, jobs per site, and the fraction of jobs that reached a
lower-ranked site after the gate opened.
