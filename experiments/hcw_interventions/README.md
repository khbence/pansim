# HCW interventions experiment

Question: how do NPI/vaccination configurations change healthcare-worker (HCW) infection and
absence?

- `HCI` column: % of HCWs infected or quarantined that day (prevalence, integer-truncated).
- `HCE` column: HCWs newly infected in the last 24 h (incidence).
- HCW = agent with a work location of type hospital (12) or doctor (14)
  (`flagHealthcareWorkers`, `src/runtime/simulationOrchestration.cpp:355-389`).

## This round: quarantine policy only

Target design is a 2-level factorial over 4 levers - quarantine (Q), closures (C), vaccination
(V), testing (T) - 16 scenarios total (see the commented-out entries in `scenarios.yaml`). Given
the ~1h/scenario compute cost, **this round only runs Q** (`baseline` vs `quarantine0`,
`--quarantinePolicy 0`) to validate the pipeline cheaply; later rounds uncomment more lines in
`scenarios.yaml` - no code changes needed to scale up.

| Lever | baseline (implicit) | deviation flag |
|---|---|---|
| Q (this round) | `--quarantinePolicy 3` | `--quarantinePolicy 0` |
| C | default closures file | `--closures inputConfigFiles/noRules.json` |
| V | default (vaccPerWeek.txt) | `--immunizationsPerWeek 0` |
| T | default (hospital rate 0.005) | `--testingProbabilities 0.00005,0.01,0.0005,0.0005,0.5,0.05` |

**Confound to keep in mind for later rounds**: the default closures file
(`inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json`) contains a
`TestingProb` rule that force-sets hospital testing to 0.5 from day 259 onward regardless of
`--testingProbabilities`, and it isn't gated by `--enableClosures`. So the T lever only makes a
difference for days 0-258 in any scenario that keeps the default closures file; only
`closuresNone` combinations keep it active for the full run. This is a real interaction, not a
bug - just don't attribute a null T-lever result on a non-`closuresNone` scenario to "testing rate
doesn't matter" without checking which days it was actually active.

## `baseline`

`results/baseline/` is reused from `experiments/validation_1yr/results/sim/` (identical
`PARAM_SET=validated` config, no overrides) rather than re-run: runs 01-19 copied verbatim, run 20
(which crashed in the original batch) regenerated on the HPC machine with the identical command,
`all_runs.tsv` rebuilt from all 20 - already complete, nothing more to do here.

## Running

`run_scenarios.sh` itself only needs a plain `python3` (no packages - it parses `scenarios.yaml`
with a small stdlib-only parser, not `pyyaml`, since HPC nodes often have no pip/network access).
`analyze.py` needs `pandas`, `matplotlib`, `scipy` (`pip install pandas matplotlib scipy`) -
commonly run on a different machine than the simulations, see below.

1. Build `build/panSim` (see `../../SETUP_HPC.md`), then on the **same machine/build** as
   `baseline` (see caveat below) run `./run_scenarios.sh` - runs any scenario in `scenarios.yaml`
   that doesn't already have `results/<name>/all_runs.tsv` (`FORCE=1` to re-run one anyway); with
   `baseline` already complete, this runs only `quarantine0` (~1h).
2. `python analyze.py` -> `outputs/per_run.csv`, `outputs/summary.csv`,
   `outputs/significance.csv`, `outputs/{hci,hce}.png`.

**Run new scenarios on the same machine/build as `baseline`** (the HPC machine, per
`SETUP_HPC.md`) - a different build/platform is a possible confound in itself, on top of whatever
NPI effect is being measured.

**Smoke test first** (a couple of minutes instead of ~1h): `RESULTS_ROOT=<scratch dir> N_RUNS=2
WEEKS=2 ./run_scenarios.sh` (runs every scenario currently in `scenarios.yaml` at tiny scale, into
a throwaway directory - doesn't touch `results/`), then `python analyze.py <scratch
dir>/results <scratch dir>/outputs` - confirms YAML parsing, flag composition (check the
scratch run's `config.txt` `cmd:` line has the override flag), `all_runs.tsv` shape, and that
`analyze.py` runs cleanly on tiny/degenerate data, before committing to the full run.

## Significance testing (`outputs/significance.csv`)

For each non-baseline scenario, each of `mean_hci`, `mean_hce`, `hce_hci_ratio`, `peak_hci`,
`hcw_infections`, `days_above_thr` (all per-run summary statistics, defined in
`hcw/metrics.py:per_run`) is compared against baseline's same-metric values across the 20 runs
each, via a two-sided Mann-Whitney U test (`hcw/metrics.py:mann_whitney_vs_baseline`) plus a
rank-biserial effect size (positive = scenario stochastically larger than baseline; sign verified
against synthetic data, see that function's docstring). `hce_hci_ratio` = a run's total new HCW
infections divided by its mean HCI (an incidence-per-unit-of-average-prevalence measure, not a
day-by-day ratio - HCI is an integer percentage that's frequently exactly 0, which would make a
per-day ratio undefined most days); it's `NaN` for a run with near-zero HCI throughout, and such
runs are dropped (with the reduced N visible in `n_baseline`/`n_scenario`) rather than raising.
P-values are corrected for multiple comparisons two ways: `q_within_metric` (BH-FDR across that
metric's scenario comparisons - the primary column, "which scenarios differ significantly in this
metric") and `q_global` (BH-FDR across every row at once, stricter, supplementary).
`significant_q05` = `q_within_metric < 0.05`.

## Extending to more levers

Uncomment the relevant line(s) in `scenarios.yaml` (all 16 factorial combinations are already
spelled out there, commented) and re-run `./run_scenarios.sh` - it skips scenarios that already
have results, so only the newly-uncommented ones get simulated.
