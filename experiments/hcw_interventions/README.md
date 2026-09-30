# HCW interventions experiment

Question: how do NPI/vaccination configurations change healthcare-worker (HCW) infection and
absence?

- `HCI` column: % of HCWs infected or quarantined that day (prevalence, integer-truncated).
- `HCE` column: HCWs newly infected in the last 24 h (incidence).
- HCW = agent with a work location of type hospital (12) or doctor (14)
  (`flagHealthcareWorkers`, `src/runtime/simulationOrchestration.cpp:355-389`).

## Scenarios so far: the 4 single levers, plus one HCW-priority isolation

Target design is a 2-level factorial over 4 levers - quarantine (Q), closures (C), vaccination
(V), testing (T) - 16 scenarios total (see the commented-out combination entries in
`scenarios.yaml`). Given the ~1h/scenario compute cost, rounds so far have run all **4 single-lever**
scenarios (`baseline` vs one lever flipped each) to see each lever's own main effect before mixing
them, plus a 5th scenario (`noVaccHCW`) outside the Q/C/V/T grid, added specifically to isolate the
HCW-vaccination-*priority* effect (see below) - later rounds uncomment more combination lines in
`scenarios.yaml` - no code changes needed to scale up.

| Lever | baseline (implicit) | deviation flag |
|---|---|---|
| Q | `--quarantinePolicy 3` | `--quarantinePolicy 0` (`quarantine0`) |
| C | default closures file | `--closures inputConfigFiles/noRules.json` (`closuresNone`) |
| V | default (vaccPerWeek.txt) | `--immunizationsPerWeek 0` (`noVacc`) |
| T | default (hospital rate 0.005) | `--testingProbabilities 0.00005,0.01,0.0005,0.0005,0.5,0.05` (`hiTest`) |

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
2. `python analyze.py` -> `outputs/per_run.csv`, `outputs/summary.csv`, `outputs/significance.csv`,
   `outputs/relative_risk_table.csv`, `outputs/priority_isolation_test.csv`,
   `outputs/{hci,hce,vulnerability}.png`.

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
`hcw_infections`, `days_above_thr`, `prevalence_RR`, `incidence_RR`
(all per-run summary statistics, defined in `hcw/metrics.py:per_run`) is compared against
baseline's same-metric values across the 20 runs each, via a two-sided Mann-Whitney U test
(`hcw/metrics.py:mann_whitney_vs_baseline`) plus a rank-biserial effect size (positive = scenario
stochastically larger than baseline; sign verified against synthetic data, see that function's
docstring). `hce_hci_ratio` = a run's total new HCW infections divided by its mean HCI (an
incidence-per-unit-of-average-prevalence measure, not a day-by-day ratio - HCI is an integer
percentage that's frequently exactly 0, which would make a per-day ratio undefined most days);
it's `NaN` for a run with near-zero HCI throughout, and such runs are dropped (with the reduced N
visible in `n_baseline`/`n_scenario`) rather than raising. P-values are corrected for multiple
comparisons two ways: `q_within_metric` (BH-FDR across that metric's scenario comparisons - the
primary column, "which scenarios differ significantly in this metric") and `q_global` (BH-FDR
across every row at once, stricter, supplementary). `significant_q05` = `q_within_metric < 0.05`.

## Is HCW a vulnerable group? (`outputs/vulnerability.png`, `prevalence_RR`/`incidence_RR`)

panSim's population-wide columns (`S`, `E`, `I1`.., `Q`, `NI`) count *every* agent, healthcare
workers included - so naively comparing HCI/HCE against those raw columns compares HCWs to a
group that still contains those same HCWs (double-counting). `hcw/metrics.py` fixes this by
building a genuinely separate "rest of population" (HCWs excluded) to compare against, then
computes a proper per-capita **relative risk**: RR > 1 means HCWs are disproportionately
affected, RR < 1 means less so, RR = 1 means no difference from the rest of the population.

`HEALTHCARE_WORKER_COUNT` (10,136) and `TOTAL_POPULATION` (178,746) are exact constants for the
validated dataset, computed once by `compute_population_constants.py` (replicates
`flagHealthcareWorkers()` exactly against the real `agents1.json`/`locations0.json`, including
that `--disableTourists` defaults to on - agent typeID 9 never actually enters the simulation).
Two ratios, both in `per_run`/`significance_table`:
- `incidence_RR` = `(mean_hce / HEALTHCARE_WORKER_COUNT) / (mean_ni_nonHCW / NON_HCW_POPULATION)`
  - both sides are **exact** new-infection-per-capita rates (`HCE` and `NI - HCE`, no quarantine
    involved on either side) - **this is the one to trust**.
- `prevalence_RR` = `mean_hci / mean_pi_nonHCW` - HCI is "infected OR quarantined", and
  `PI_nonHCW` (the rest-of-population analog) inherits the same definition, so a quarantine-policy
  change can move this ratio for reasons that have nothing to do with actual infection risk (e.g.
  `quarantine0` flips `prevalence_RR` from 0.51 to 1.14, mostly because removing quarantine
  removes a large "quarantined-but-not-infected" contribution from the population side, not
  because HCWs' own risk changed that much) - treat it with caution, see `add_population_columns`'s
  docstring for exactly how it's built (including the one approximation involved: HCI is an
  integer-truncated percentage, so recovering an HCW infected-or-quarantined *count* from it
  needs a small de-truncation step; `incidence_RR` needs no such approximation, `HCE` already
  being an exact count).

`outputs/vulnerability.png` plots both, one row per scenario, prevalence and incidence side by
side, HCWs vs rest-of-population - on the real data, `incidence_RR` shows HCWs at 1.38x
(baseline) to 1.47x (`quarantine0`) the rest-of-population's daily new-infection rate: a
consistent, not-quarantine-confounded signal that HCWs are a vulnerable group in both scenarios.

### Why `prevalence_RR` and `incidence_RR` disagree: hospitals are exempt from workplace quarantine

`prevalence_RR` swings from 0.51 (baseline) to 1.14 (`quarantine0`) while `incidence_RR` stays
steady around 1.4 in both. This isn't noise - it's a real, traceable mechanism in the quarantine
algorithm itself, not an artifact of the analysis:

`quarantineAgent()` (`include/policies/movementPolicies.h:179-248`), called whenever an agent is
diagnosed, escalates by `quarantinePolicy` level:
```
policy == 0        -> returns immediately, nothing happens
policy >= 1         -> the diagnosed agent itself is quarantined
policy > 1  (Q2+)   -> + their HOME is quarantined (locationQuarantineUntilPtr[home] = until)
policy > 2  (Q3+)   -> + their WORK/CLASSROOM location is quarantined - but see below
```

The work-quarantine step (line 236) resolves the agent's work-slot location, then checks:
```cpp
if (loc != MAX
    && (a.locationTypePtr[loc] == a.workType || a.locationTypePtr[loc] == a.classroomType)
    ...) {
    a.locationQuarantineUntilPtr[loc] = until;   // close the location
}
```
`a.locationTypePtr[loc]` is the location's *true* type (from `locations0.json`), not just "this is
someone's work slot". For a healthcare worker, the location assigned to their work slot has true
type 12 (hospital) or 14 (doctor) - the exact same check `flagHealthcareWorkers()` uses to call
them a HCW in the first place. So `locationTypePtr[loc] == workType` is **never true** for a
HCW's workplace, and hospitals/doctors' offices are **never quarantined/closed** via this
mechanism, at any quarantine policy level. The function's own commented-out lines confirm this is
intentional, not an oversight:
```cpp
// only quarantining work & classrooms
//&& a.locationTypePtr[loc] != a.doctorType
//&& a.locationTypePtr[loc] != a.hospitalType
```
(i.e. hospitals used to be explicitly excluded; now the positive `== workType` check already
excludes them, and the exclusion lines are dead but left as a record of the intent - hospitals
have to keep functioning during an epidemic, so a diagnosed staff member can't shut the ward down
the way it would shut down a regular office.)

Home quarantine (Q2) has no such exemption - a HCW's household is quarantined exactly like
anyone else's if they or a family member is diagnosed.

**Consequence**: at baseline (Q3), a regular workplace closing every time a diagnosed coworker is
found there sends many coworkers into quarantine at once (a compounding effect across a whole
workforce) - this is what drives the general population's `PI_nonHCW` (infected-or-quarantined)
so high in the second wave (see `outputs/vulnerability.png`'s baseline-prevalence panel). Hospitals
never trigger this channel for their own staff, so `HCI` stays comparatively low - not because HCWs
are safer, but because one whole quarantine pathway structurally doesn't apply to their workplace.
This is exactly why `prevalence_RR` (which inherits HCI's infected-OR-quarantined definition on
both sides) swings so much with the quarantine policy, while `incidence_RR` (pure infection,
unaffected by which quarantine channels exist) stays stable and is the metric to trust for the
"is HCW a vulnerable group" question.

### Isolating the HCW-priority effect (`noVaccHCW`)

`noVacc` (`--immunizationsPerWeek 0`) removes vaccination for the *entire population*, so its
`incidence_RR` increase (baseline 1.38 -> 1.78, the largest of the 4 single levers) conflates two
things: "vaccination exists at all" and "HCWs specifically get it first". `noVaccHCW` isolates the
second: `--immunizationOrder 0,2,3,4,5,6,0,0,7,8` changes only the first value of the 10-category
order string (category 0 = healthcare workers, per the `--immunizationOrder` help text order:
health workers, nursery home, 60+, 18-60+underlying, essential worker, 18+, 60+underlying, school
teachers, 12-18, 5-12) from `1` (first priority) to `0` (skip entirely) - every other category's
order and `vaccinationGroupLevel` is untouched, so the *rest of the population's* vaccination
rollout is unaffected (confirmed safe by reading the scheduling code,
`include/immunization.h:495-516`: categories are found by searching for `round == i+1` and the
loop just moves on when a round is empty, so HCWs' unclaimed doses flow to the next eligible
category the same day - no wasted supply, no timing shift for anyone else; `0` as a skip value is
already used twice in the default order for other categories, so this isn't an edge case).

Comparing `noVaccHCW` against both `baseline` and `noVacc` answers the actual question: is it
population-level vaccination coverage (indirect/herd protection) or HCWs' own vaccination access
specifically that matters for their elevated risk?

**Result** (`outputs/priority_isolation_test.csv`, direct `noVaccHCW` vs `noVacc` test, not vs
`baseline`): `incidence_RR` is 1.78 under `noVacc` (nobody vaccinated) vs **1.82 under `noVaccHCW`**
(everyone *except* HCWs vaccinated) - i.e. excluding *only* HCWs from vaccination leaves them
*slightly worse off* than removing vaccination from the entire population (median diff +0.046,
Mann-Whitney p=0.0008, significant but a much smaller effect than any of the single-lever-vs-baseline
comparisons - rank-biserial 0.62 vs 1.0 elsewhere). So it's **not** primarily herd protection from
the rest of the population being vaccinated that shields HCWs - their own vaccination access is
what matters, and losing it isn't meaningfully cushioned by everyone else around them being immune.
This directly supports treating HCW vaccination priority as the standalone protective measure it's
designed to be, not as something substitutable by population-level coverage.

## Extending to more levers

Uncomment the relevant line(s) in `scenarios.yaml` (all 16 factorial combinations are already
spelled out there, commented) and re-run `./run_scenarios.sh` - it skips scenarios that already
have results, so only the newly-uncommented ones get simulated.
