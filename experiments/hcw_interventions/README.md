# HCW interventions experiment

*(A Hungarian-language summary of the results and their evaluation is in `README.hu.md`.)*

Question: how do NPI/vaccination configurations change healthcare-worker (HCW) infection and
absence?

- `HCI` column: % of HCWs infected or quarantined that day (prevalence, integer-truncated).
- `HCE` column: HCWs newly infected in the last 24 h (incidence).
- HCW = agent with a work location of type hospital (12) or doctor (14)
  (`flagHealthcareWorkers`, `src/runtime/simulationOrchestration.cpp:355-389`).

## Scenarios so far: 4 single levers, 1 priority-isolation scenario, 4 two-way combinations, 3 round-3 policies

Target design is a 2-level factorial over 4 levers - quarantine (Q), closures (C), vaccination
(V), testing (T) - 16 scenarios total (see the commented-out combination entries in
`scenarios.yaml`). 13 scenarios have been run so far: `baseline`; the 4 single levers
(`quarantine0`, `closuresNone`, `noVacc`, `hiTest`); `noVaccHCW` (outside the Q/C/V/T grid, isolates
the HCW-vaccination-*priority* effect specifically, see below); 4 two-way combinations
(`noVaccHCW_hiTest`, `closuresNone_hiTest`, `quarantine0_noVaccHCW`, `quarantine0_hiTest`) chosen
to probe interactions between the levers found to matter most; and 3 "round 3" policies from
professor feedback (`hcwDailyTest`, `longQuarantine`, `hcwPPE` - see "New intervention ideas"
below for the brainstormed list and implementation details; `hcwPPE` needed a C++ code change).
See "Full results and interaction effects" below for the complete picture - **`hcwPPE` is now the
strongest result in the entire series** (RR=0.71, a genuine reversal of the vulnerability gap, not
just a reduction). Later rounds can uncomment more combination lines in `scenarios.yaml` - no code
changes needed to scale those up.

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

### Does more testing compensate for lost vaccine priority? (`noVaccHCW_hiTest`)

The two dominant single levers combined: `noVaccHCW` (HCWs excluded from vaccination) +
`hiTest` (hospital testing rate 0.005 -> 0.5). `incidence_RR`: 1.38 (baseline) -> 1.82
(`noVaccHCW` alone, the worst single lever) -> 1.09 (`hiTest` alone, the only protective lever) ->
**1.52 (`noVaccHCW_hiTest`, combined)**.

Checking against a purely additive model (no interaction): baseline + `noVaccHCW`'s effect
(+0.445) + `hiTest`'s effect (-0.285) predicts 1.538 - within 0.02 of the observed 1.520. **The two
levers are close to additive, with no evidence of a strong interaction** (testing doesn't act
*through* the vaccine-access channel, or vice versa - they look like two independent, stacking
layers of protection, not substitutes that cancel each other's mechanism out).

In absolute terms: losing HCW vaccine priority alone adds +0.445 to `incidence_RR` above baseline;
with high testing added on top, that excess drops to +0.142 - **testing recovers about 68% of the
excess risk caused by losing vaccine priority, but not all of it**. `noVaccHCW_hiTest`'s RR (1.52)
is still significantly above baseline (1.38), so testing is a strong complement to vaccine
priority, not a full substitute. (Caveat: the T lever here only differs from baseline for days
0-258 - see the confound note at the top - so this 68% figure reflects testing's effect during
that window of the first wave; its value during the untested second wave, where the closures file's
own TestingProb rule already forces testing to 0.5 regardless, isn't isolated by this comparison.)

## Full results and interaction effects

Complete `outputs/relative_risk_table.csv` (`incidence_RR` is the trustworthy metric throughout -
`prevalence_RR` carries the quarantine/HCI-definition caveat from above):

| scenario | incidence_RR | vs baseline | significant? |
|---|---|---|---|
| baseline | 1.38 | - | - |
| **hcwPPE** | **0.71** | **-0.67** | **yes (RR itself != 1 too, p=1.9e-6 - see below)** |
| hcwDailyTest | 1.00 | -0.38 | yes (but RR itself not != 1, see below) |
| closuresNone | 1.38 | +0.00 | no |
| hiTest | 1.09 | -0.29 | yes |
| closuresNone_hiTest | 1.13 | -0.25 | yes |
| longQuarantine | 1.35 | -0.03 | no |
| quarantine0 | 1.47 | +0.09 | yes |
| quarantine0_hiTest | 1.48 | +0.10 | yes |
| noVacc | 1.78 | +0.40 | yes |
| noVaccHCW | 1.82 | +0.44 | yes |
| noVaccHCW_hiTest | 1.52 | +0.14 | yes |
| quarantine0_noVaccHCW | 1.62 | +0.25 | yes |

### Hospital-specific PPE reverses the gap entirely (`hcwPPE`) - the strongest result of the series

`hcwPPE` applies a 70% infectiousness reduction (coefficient 0.3) at hospital/doctor locations
only, active from day 0 - see "New intervention ideas" below for the full implementation (it
needed a new C++ closure rule, `HospitalPPE`, mirroring `Masks` but targeted at
`hospitalType`/`doctorType` instead of every non-home location).

**Result: `incidence_RR` = 0.71 - not just parity, but a genuine reversal.** Every single one of
the 20 runs has `incidence_RR < 1` (median 0.713, range 0.67-0.77); a one-sample Wilcoxon test
against the null of RR=1 gives p=1.9x10^-6 (the maximum possible significance for n=20) -
**healthcare workers become measurably *less* at risk than the rest of the population.** This is
visible directly in `outputs/vulnerability.png`'s `hcwPPE` incidence panel - the blue (HCW) line
sits clearly *below* the orange (population) line at both peaks, the only scenario where that
happens.

This is now the single strongest result in the whole experiment series: the largest effect size
measured (-0.67 vs baseline, nearly double `hcwDailyTest`'s already-large -0.38) and the only
intervention that doesn't just close the vulnerability gap but inverts it. Makes sense
mechanistically - PPE acts directly on transmission *at the exposure site itself*, independent of
the detect-then-isolate pathway that `hcwDailyTest`/`hiTest` depend on (and that `quarantine0_hiTest`
showed goes to zero without active quarantine) - it doesn't compete with or depend on any of the
other levers tested, it's a structurally different, more direct protective mechanism.

### Mandatory daily testing closes the gap entirely (`hcwDailyTest`)

`hiTest` raised the testing *probability* (0.005 -> 0.5) but left `--testingRepeatDelay` at its
default 5 days, so even a "daily-tested" agent could in practice only be *re*-tested every 5 days.
`hcwDailyTest` fixes both (probability 0.95, `testingRepeatDelay=1`) - a true "test every HCW every
morning" policy.

**Result: `incidence_RR` = 1.00 (median 0.9966) - parity with the rest of the population.** A
one-sample test against the null of RR=1 (Wilcoxon signed-rank on the 20 per-run values) gives
p=0.67 - **not significantly different from 1** (11 of 20 runs land below 1, 9 above - a coin-flip
split, not a one-sided trend). This is visible directly in `outputs/vulnerability.png`'s
`hcwDailyTest` incidence panel - the two lines track each other throughout, unlike most other
scenarios.

Before `hcwPPE` was run, this was the strongest result of the series; it's still the second
strongest and the best *testing-based* result - a large protective effect (-0.38 vs baseline,
larger than `hiTest`'s already-substantial -0.29) that fully closes the gap rather than narrowing
it. Consistent with the mechanistic finding that testing works through the quarantine/isolation
channel (see "the key mechanistic finding" below) - removing the retest-delay bottleneck lets
genuinely daily screening catch and isolate infected HCWs almost immediately, maximizing that
channel's effect. Unlike `hcwPPE`, though, this protection is conditional on quarantine being
active at all (see `quarantine0_hiTest` below) - `hcwPPE`'s effect has no such dependency.

### Does quarantine *duration* matter, independent of whether it triggers at all? (`longQuarantine`)

Extending isolation from the default 10 days to 21 (`--quarantineLength 21`, `quarantine0` itself
unchanged at policy level 3) reduces absolute HCW infections significantly (`mean_hce`: -0.68,
`hcw_infections`: -249, both p<0.0001) - longer isolation does reduce transmission. But
**`incidence_RR` barely moves (1.38 -> 1.35) and the change is not significant** (p=0.118, in
contrast to every other lever tested so far). Longer isolation reduces infections for HCWs and the
rest of the population roughly proportionally - like `closuresNone`, it's a lever that moves
everyone together rather than changing HCWs' *relative* position. (`prevalence_RR` does move
slightly, -0.02, p=0.016 - consistent with the established mechanism: longer quarantine duration
inflates the general population's "quarantined" pool specifically, since they're the ones who
benefit from workplace quarantine - a small version of the same effect seen with `quarantine0`.)

### Does the testing effect hold up unconfounded? (`closuresNone_hiTest`)

This scenario removes the default closures file's day-259 `TestingProb` override (so testing's
effect is measured for the full year, not just days 0-258) - but it turns out **this doesn't
cleanly isolate testing's "true" effect**, because `closuresNone` doesn't *only* remove that one
override: it replaces the closures file with `noRules.json` (`{"rules": []}`), which also removes
every `ExposeToMutation` event (no new variants ever get seeded - the whole year runs on the
initial variant only, no second wave) *and* every other closure rule (masks, curfew, school
restriction). That's a much bigger change than "fix the testing confound" - it was flagged as a
risk back when `closuresNone` was first proposed.

Result: `closuresNone_hiTest`'s `incidence_RR` is 1.13 (-0.25 vs baseline), very close to confounded
`hiTest`'s 1.09 (-0.29) - similar magnitude, if anything *slightly* smaller. This is reassuring
(the day-259 cutoff wasn't masking a much larger effect), but it isn't a clean "unconfounded
testing effect" measurement, because the variant-seeding and other-closures changes are entangled
with it. A properly isolated test would need a custom closures file that keeps variant seeding but
drops only the `TestingProb` override - not built here.

### Worst case: stacking the two harmful levers (`quarantine0_noVaccHCW`)

Naive additive prediction: baseline + `quarantine0`'s effect (+0.09) + `noVaccHCW`'s effect (+0.45)
= 1.91. Observed: **1.62** - a large (-0.29) deviation from additive, in sharp contrast to
`noVaccHCW_hiTest`'s near-perfect additivity (-0.02 deviation). The two harmful levers do **not**
simply stack.

Direct pairwise checks (`outputs/per_run.csv`, Mann-Whitney) pin down why:
- Adding `noVaccHCW` on top of `quarantine0` (`quarantine0` -> `quarantine0_noVaccHCW`): RR rises
  +0.155, significant (p<1e-7) - smaller than the +0.445 it adds on top of plain `baseline`, but
  still clearly harmful and in the expected direction.
- Adding `quarantine0` on top of `noVaccHCW` (`noVaccHCW` -> `quarantine0_noVaccHCW`): RR
  **drops** -0.199, significant (p<1e-7) - the opposite sign from `quarantine0`'s own effect vs
  baseline (+0.09)!

This reversal traces back to the same mechanism as the prevalence/incidence disagreement: hospitals
are structurally exempt from workplace quarantine (see above), so HCWs were never getting much
protection from that specific channel to begin with - the **rest of the population** was. Removing
quarantine (`quarantine0`) therefore raises the rest-of-population's infection rate proportionally
*more* than HCWs': checking the underlying absolute rates (median `mean_hce` / `mean_ni_nonHCW`)
going from `noVaccHCW` to `quarantine0_noVaccHCW`, HCW incidence rises **24%** (16.7 -> 20.8) while
the rest-of-population's rises **39%** (152.3 -> 212.4) - the denominator catches up faster than
the numerator, narrowing the ratio even though both groups get strictly more infections in absolute
terms. Sub-additivity in a shared, finite susceptible population (diminishing marginal harm as
infections saturate) likely compounds this further.

### Does testing compensate for lost quarantine too? (`quarantine0_hiTest`) - the key mechanistic finding

This is the most important result of the three new scenarios. `quarantine0_hiTest`'s `incidence_RR`
(1.477) is statistically indistinguishable from `quarantine0` alone (1.469) - direct Mann-Whitney
test, p=0.51, **not significant** (vs. `noVaccHCW_hiTest`'s rank-biserial effect of 1.0 when tested
against `noVaccHCW`). **Adding high testing provides essentially zero benefit when quarantine is
already disabled**, in stark contrast to the ~68% harm-recovery testing provided for lost vaccine
priority.

This makes complete mechanistic sense once traced through the code: testing's entire protective
mechanism is *detect, then isolate* - a positive test result only changes anything if a positive
agent is actually quarantined as a result. `quarantineAgent()` (`include/policies/movementPolicies.h:181`)
returns **immediately** if `a.quarantinePolicy == 0`, before anything else happens - so under
`quarantine0`, a positive test changes nothing about that agent's behavior at all. Testing is only
protective *through* the quarantine pathway; it has no independent transmission-reducing mechanism
of its own in this model. `noVaccHCW_hiTest` worked because that scenario keeps quarantine active
(policy 3, baseline default) - testing could still trigger real isolation there.

**This reframes the whole ranking**: testing isn't an independent "4th lever" on equal footing with
quarantine, vaccination and closures - it's a *multiplier* on whatever quarantine policy is already
in place. "Increase testing" is good policy advice only conditional on "and quarantine/isolation
infrastructure exists to act on the result." Quarantine's own modest direct effect (+0.09 removing
it) therefore understates its true importance - it's also a **prerequisite** for testing to have
any teeth at all.

### Updated ranking of what matters for HCW vulnerability

0. **Hospital-specific PPE (`hcwPPE`)** - the single most effective intervention found, by a wide
   margin: a 70%-effectiveness assumption at hospital/doctor locations doesn't just close the
   vulnerability gap, it **reverses** it (RR=0.71, every one of 20 runs below 1, p=1.9e-6).
   Mechanistically distinct from every other lever tested - it protects at the exposure site
   directly, with no dependency on quarantine/testing/vaccination infrastructure being in place.
   The clearest single real-world policy recommendation this experiment series has produced.
0b. **Testing intensity/frequency, done properly (`hcwDailyTest`)** - the best *testing-based*
   result: fixing the retest-delay bottleneck that `hiTest` left in place turns testing from
   "substantially helps" (-0.29) into "closes the gap entirely" (RR=1.00, not significantly
   different from parity). Still conditional on quarantine being active (see #2), unlike PPE -
   this is the ceiling of what testing can achieve when that condition holds.
1. **Vaccination (direct HCW access)** - the largest, most robust *harmful-when-removed* effect in
   every context tested (+0.40 to +0.45), and the one lever shown to matter for HCWs
   *specifically*, not just via population-level herd effects (see "Isolating the HCW-priority
   effect" above).
2. **Testing, conditional on quarantine being active** - large and reliable when quarantine policy
   is in place (-0.29 to -0.38 depending on how aggressively it's implemented, recovers ~68% of
   lost-vaccine-priority harm at the `hiTest` level and the entire gap at the `hcwDailyTest` level),
   but **worthless on its own** if quarantine is disabled (p=0.51, no effect). Not a standalone
   lever - a quarantine-dependent one.
3. **Quarantine policy** - a modest *direct* effect from whether it's enabled at all (+0.09 when
   fully removed) that undersells its real importance, since it's also the prerequisite for
   testing's much larger effect to materialize. *Duration* alone (10 -> 21 days, `longQuarantine`),
   independent of whether it triggers, does **not** significantly change `incidence_RR` (p=0.12) -
   it's whether quarantine exists at all that matters for the ratio, not how long it lasts once
   triggered. Also the lever whose *removal* paradoxically narrows the HCW-vs-population gap when
   vaccine priority is already lost (see "Worst case" above) - a reminder that these levers
   interact with each other, not just with a fixed baseline.
4. **Closures (masks/curfew/school-restriction/venue-closures bundle)** - no measurable effect on
   the HCW-vs-population *ratio* in any test so far (closuresNone alone: not significant;
   closuresNone_hiTest: similar magnitude to hiTest alone). Likely because closures act on
   transmission broadly rather than differentially by occupation - they move both groups together,
   the same pattern seen with quarantine *duration* above.

## Extending to more levers

Uncomment the relevant line(s) in `scenarios.yaml` (all 16 factorial combinations are already
spelled out there, commented) and re-run `./run_scenarios.sh` - it skips scenarios that already
have results, so only the newly-uncommented ones get simulated.

## New intervention ideas (professor feedback)

Feedback round asked for more realistic, real-world HCW-specific policies beyond the Q/C/V/T
factorial. Brainstormed list below, split by whether panSim could already express them.

### Ready to test with existing flags (no code change)

- **`hcwDailyTest`** (run - see "Mandatory daily testing closes the gap entirely" above) -
  mandatory daily HCW testing. `hiTest` alone only raised the probability (0.5); it left
  `--testingRepeatDelay` at its default 5 days, so even a "daily-tested" agent could in practice
  only be *re*-tested every 5 days. True daily testing needs both:
  `--testingProbabilities 0.00005,0.01,0.0005,0.0005,0.95,0.05 --testingRepeatDelay 1`. **Result:
  the strongest finding of the whole series - RR reaches parity (1.00), not just reduced.**
- **`longQuarantine`** (run - see "Does quarantine duration matter" above) - extends isolation
  duration from the default 10 days (`--quarantineLength`, registered once by `RealMovement`,
  `include/policies/movementPolicies.h:1401`) to 21, independent of whether quarantine triggers at
  all (that's what `quarantine0` tested). **Result: no significant change to `incidence_RR`** -
  duration alone doesn't move the ratio, only whether quarantine exists at all does.
- **`hcwDailyAntigenTest`** (not yet run) - `hcwDailyTest`'s flags plus `--testingMethod antigen`
  instead of the default PCR - models realistic rapid-antigen screening (faster/cheaper/more
  frequent in reality, typically less sensitive).
- **`hcwFullVacc`** (not yet run) - push HCW vaccination coverage (`--vaccinationGroupLevel`'s
  first value, default 0.9) to 0.99. Complements `noVaccHCW` (0%) by testing the other direction -
  is there a ceiling effect above today's 90%?
- **`earlyVacc`** (not yet run) - `--immunizationStart 30` instead of the default 96. Since HCWs
  are already first-priority, starting the whole rollout earlier benefits them soonest and most -
  models fast-tracked vaccine procurement/approval.

### Hospital-specific PPE (`hcwPPE`) - implemented via a C++ code change, run - see "Hospital-specific
PPE reverses the gap entirely" above for the result (RR=0.71, the strongest finding of the series)

The single most realistic, most-studied real-world HCW protection (N95 masks, gowns, eye
protection) had no equivalent lever: the existing `"Masks"` closure rule
(`include/policies/closurePolicies.h:214-246`) scales infectiousness at **every non-home location
type uniformly** (`if (type != homeType)`) - there was no way to target hospitals specifically.

Added a new `"HospitalPPE"` closure rule (`closurePolicies.h`, right after the `Masks` block it
mirrors) that scales infectiousness only where `type == hospitalType || type == doctorType`,
using the exact same toggle mechanics as `Masks` (a `Rule` + `GlobalCondition`, state tracked via
`previousOpenState`, multiply-on-activate/divide-on-revert). Also added a `currentPPEValue` field
to `Simulation` (`include/simulation.h`) for symmetry with `currentMaskValue` - not wired into the
older `processFlags()` string-flag interface, which this experiment doesn't use.

`inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8_hospitalPPE.json` is a copy of
the validated closures file with one rule appended:
```json
{"name": "HospitalPPE", "parameter": 0.3, "conditionType": "afterDays",
 "threshold": 0, "threshold2": 0, "openAfter": 10000, "closeAfter": -1, "locationTypesToClose": []}
```
`threshold: 0, openAfter: 10000` makes it active from day 0 for effectively the whole run (any
run far shorter than 10000 days). `parameter: 0.3` means infectiousness at hospital/doctor
locations is multiplied by 0.3 once PPE activates - i.e. a **70% reduction**, a middle estimate
from the range reported for N95/eye-protection effectiveness in healthcare settings in the
literature (e.g. Chu et al. 2020, *Lancet*, physical distancing/face protection meta-analysis) -
real-world effectiveness varies substantially by compliance and PPE type, so treat this as an
illustrative, documented modeling assumption, not a precisely validated epidemiological parameter.

**Verified before trusting it**: rebuilt (`make buildCPU`), then a 1-week, 1-run smoke test with
`--diags 1` confirmed the rule fires exactly once, on day 0, with the right coefficient -
`HospitalPPE on with 0.3 multiplier` appears right after the day-0 data row (note: like all of
this codebase's rule diagnostics, e.g. `Masks`, the `printf` goes to stdout, so it's interleaved
with the `.tsv` data rows, not the `.log` file - this is pre-existing behavior, not specific to
`HospitalPPE`). The filter logic itself (`type == hospitalType || type == doctorType`) is a direct,
minimal change from `Masks`'s already-proven-correct pattern, so this smoke test plus a careful
code read were judged sufficient without also dumping raw per-location infectiousness arrays.

### Ideas that need more code work than this round covered

- **COVID-ward cohorting** (separating infectious patients from the general hospital population to
  reduce ambient HCW exposure) - panSim has only one "hospital" location type, no COVID-ward vs.
  general-ward distinction. Would need new location-type granularity.
- **HCW-specific quarantine policy independent of the population-wide policy** - given the
  hospital-workplace-quarantine exemption finding above, a natural policy is "quarantine can't
  close a hospital, but a diagnosed HCW should still be pulled from the roster faster/longer than
  the general population." `--quarantinePolicy`/`--quarantineLength` are currently single global
  values, not per-category.
- **Reduced shift/contact-hour scheduling for HCWs during high-prevalence periods** - no existing
  mechanism; would need new schedule-modification logic analogous to `ReduceMovement`'s
  `stayHome` mechanic but targeted at the HCW category specifically (today `ReduceMovement`
  applies to a population fraction, not a named category).
