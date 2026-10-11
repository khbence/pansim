# ACT-049 phase 3: response to 10, 11 and 12, and the runs they asked for

11 October 2026. This note answers `10_reading_of_09_phase2.md`, `11_volume_model_proposal.md` and `12_capacity_work_package (1).md`. Code is in `superspreader/study/` (`run_phase3.py`, `run_phase3b.py`, `chain_structure.py`, `capacity_audit.py`, `capacity_recalc.py`, `analyze_phase3.py`) and in the PanSim sources (`--volumeModel`, `--volumeTypeFactors`, `--volumeHomeType`, `--occupancyStats`, and a `Ventilation` closure rule). Summaries are in `study/summaries/p3_*.json` and `p2_*.json`.

## 1. What I accepted from 10, what I changed, and what I did not do

Accepted and done:

- **CV² column.** The column in the 09 grid table was the squared coefficient of variation of realised offspring (`1/R + 1/k`), so it included Poisson noise. 09 now reports `1/k`, the quantity that adds across independent sources.
- **Caregiving tail (10 section 4).** Checked from traces, with ten times more chains (section 2).
- **More chains, more wave-2 replicates, Set II at other strengths** (sections 2, 3, 4).
- **Doc edits.** Status notes added to `02_brief.md` and `03_pansim_options_table.md`.

Changed or not done:

- **Wave-2 verdict in 10 section 5** used the 4-replicate baseline of 222. With 8 baseline replicates the baseline wave-2 peak is 227 (SD 35), and the baseline wave-1 hospital area is 10,187, not the 8,565 target the corners were calibrated to. Every corner was therefore fitted about 15% below the larger baseline sample. Section 3 states the results against the 8-replicate baseline.
- **The occupancy cap lever** (12 E7) was not implemented. An occupancy cap needs the movement code to redirect refused agents, which is a larger change than a hazard-side lever and has no clean definition in a model whose venues are aggregates of thousands of sampled visitors. The ventilation lever is implemented (a `Ventilation` closure rule that scales the hazard of the listed location types).
- **Design capacity (B1) could not be reconstructed.** Section 5 explains why, and what I used instead.

## 2. The caregiving tail: setting inheritance, not a short generation interval

Setup: 40 replicates per arm (about 38,600 chains each), about 1000 seeds per run, mean offspring matched near 0.3. All numbers from `p2_chains.json` and `p3_chain_structure.json`.

| arm | chains | mean offspring | k (all nodes) | zero share | chains > 10 | chains > 50 | max chain |
|---|---|---|---|---|---|---|---|
| flat, shape 0.04 | 38,630 | 0.294 | 0.043 | 0.914 | 417 | 17 | 101 |
| caregiving profile + shape 0.09 | 38,658 | 0.320 | 0.083 | 0.860 | 236 | 19 | 249 |

Against a negative binomial branching process with each arm's own mean and k (300,000 simulated chains per arm):

| arm | expected chains > 10 | observed | expected chains > 50 | observed |
|---|---|---|---|---|
| flat, shape 0.04 | 436 | 417 | 13 | 17 |
| caregiving + shape 0.09 | 432 | 236 | 3.9 | 19 |

- The flat arm follows its marginal. The caregiving arm does not: its middle of the tail is thinner than the negative binomial predicts (236 against 432) and its extreme is heavier (19 against 3.9 chains over 50, and a maximum of 249 against 101). This confirms the 4-replicate finding in 10 at ten times the sample, and the "0 against 3" in 09 is now 17 against 19, so the claim in 09 that only the multiplier produces long chains is withdrawn: the flat shape-0.04 arm produces them too (17 of 38,630).
- **Mechanism 2 (shorter generation interval) is rejected.** Mean realised generation interval is 6.20 days in the flat arm and 6.25 in the caregiving arm.
- **Mechanism 1 (generation-correlated setting) is supported.** The offspring of a case infected in a hospital are mostly infected in a hospital, and such cases have much higher offspring than cases infected at home:

| arm | parent infected at | cases | mean offspring | share of children infected in a hospital |
|---|---|---|---|---|
| caregiving + shape 0.09 | home | 10,539 | 0.22 | 0.17 |
| caregiving + shape 0.09 | hospital | 3,813 | 0.68 | 0.70 |
| flat, shape 0.04 | home | 5,807 | 0.25 | 0.07 |
| flat, shape 0.04 | hospital | 1,003 | 0.47 | 0.47 |

Hospital-infected cases are 21% of non-seed infections in the caregiving arm (6% in the flat arm), and 46% of the infections in its chains over 10 are in hospitals. A near-closed subpopulation with a repeated high-hazard setting produces a heavier extreme and a thinner middle than independent offspring can. This is a substantive finding: calibrating PanSim to a published k does not calibrate its large-outbreak risk, because the offspring of a case are not independent of where the case was infected.

Isolation (quarantine policy 1) at 4 replicates, as in 09, is unchanged and is not re-run at 40 replicates; the 0 against 2 chains over 50 there is within noise.

## 3. Wave-2 hold-out with more replicates

Baseline now has 8 replicates (2 original seeds 101, 102 plus 6 new), shape 0.3 has 12. Mean (SD):

| corner | replicates | wave-1 peak | wave-2 peak | wave-1 area (105 d) | total infections | Welch t against baseline wave 2 |
|---|---|---|---|---|---|---|
| baseline | 8 | 172 (19) | 227 (35) | 10,187 | 52801 | |
| multiplier shape 0.3 | 12 | 149 (32) | 215 (44) | 8,695 | 49,197 | -0.68, not different |
| multiplier shape 0.1 | 4 | 132 (41) | 152 (36) | 7,259 | 39,504 | -3.49, differs |
| covid venue strength 1 | 4 | 156 (33) | 148 (18) | 8,188 | 39,355 | -5.17, differs |
| venue strength 4 (hand-bracketed) | 4 | 177 (39) | 120 (29) | 8,318 | 34,062 | -5.58, differs |
| shape 0.3 + venue strength 1 | 4 | 154 (15) | 105 (33) | 8,530 | 33,068 | -5.93, differs |
| volume model (non-residential, peak-anchored, mid) | 4 | 154 (15) | 92 (16) | 8,131 | 27,251 | -9.20, differs |

- With 12 replicates, shape 0.3 still survives the hold-out: its wave-2 peak is 215 against 227, with no detectable difference. The survivor in 10 is confirmed at a larger sample, with the caveat that it is "not rejected" and the SD of its wave-2 peak (44) is wider than the baseline's, so power against a 15% difference is low.
- Its early k in the fitted setting is 0.27 (12 replicates). Wave-1 peak and area are 13% to 15% below the 8-replicate baseline because all corners were calibrated to the original two-seed target of 8,565.
- The volume model fails the hold-out as badly as the venue-profile corners (wave-2 peak 92, 60% below baseline, and total infections about 48% lower).

## 4. Set II lever comparison at other venue strengths

Same design as 09 section 4 (day-21 levers, 12 weeks, 4 paired replicates, calibrated to early R 1.3 on two seeds; the measured early R of the uncontrolled arm is shown). Reduction in infections relative to no lever, mean over replicates:

| lever | strength 2 | strength 4 | strength 8 (from 09) | Set I (multiplier shape 1) |
|---|---|---|---|---|
| uncontrolled early R | 1.15 | 1.19 | 1.17 | 1.30 |
| masks | 75.6% | 79.5% | 78.8% | 73.6% |
| close venue types 5, 6, 8, 9 | 68.3% | 75.4% | 75.3% | 37.2% |
| isolate diagnosed + household | 41.5% | 37.6% | 34.2% | 53.3% |
| isolate diagnosed only | 21.2% | 22.7% | 23.4% | 16.9% |

- The rank reversal (venue closure beats household isolation when the venues carry the amplification, and loses when the transmission is person-level) holds at all three strengths, with venue closure at 68% to 75% against household isolation at 34% to 42%.
- The reversal tracks where transmission occurs, not the value of k. The strength-4 arm has early k of about 3 (from 06 and 09 sweeps) and is still firmly in the venue-closure-wins regime. The Set I arm has k near 0.9. So "dispersion" does not predict the best lever; the location of the high-hazard contacts does.
- Per 1000 quarantine person-days, agent-only isolation averts about 98 to 102 infections in Set II (all three strengths), household isolation 52 to 64 at strengths 2 and 4 and 43 at strength 8. In Set I the order is reversed (household 108, agent-only 77).

## 5. Capacity audit and recalculation (12 phases A to D)

### 5.1 What `locations0.json` contains

Audit of the 80,877 locations, 184,745 agents, with an uncontrolled 3-week reference run (`--occupancyStats`, new). Files: `summaries/p3_capacity_audit_original_summary.json` and `_per_location.csv`.

- **Capacity and area are inert.** `capacity` and `area` are read into the location list and never used in a transmission or movement computation (the only reference is a pointer that is set and not dereferenced). The regression run (E0) with all new flags off reproduces the daily table and the infection trace of an earlier 8-week run exactly.
- **Area is populated only for residences (10,000 for every cell) and public spaces.** For every other type area is 0, so the schema's area column is unused.
- **Residence capacity does not follow the assigned count.** Residence locations are 100 m grid cells. Their capacity is the cell's population (median 92, sum 9.2 million, close to the population of Hungary), while the median number of sampled agents assigned is 2. The sample is therefore about 1 in 50 of the population, and the residence is a household of sampled agents drawn from a cell of about 90 residents. The assigned agents are real households (age gap within a location averages 13.6 years against 25.4 for random pairs; sizes 1 to 7, mean 2.6).
- **The other types are constant or placeholder.** 14 (health centres) all 1000, 44 (universities) all 1200, 22 (nursing homes) all 55, 15 all 1,000,000; types 5, 9, 11, 12 have two distinct values.

| type | locations | capacity median (max) | distinct values | in stated range | peak occupancy median | peak / capacity median | share of locations with peak above capacity |
|---|---|---|---|---|---|---|---|
| 2 residence | 73,036 | 92 (520) | 276 | | 3 | 0.035 | 0.006 |
| 3 school stay site | 181 | 138 (979) | 79 | | 114 | 0.90 | 0.44 |
| 4 workplace | 119 | 100 (500) | 7 | | 10 | 0.25 | 0.14 |
| 5 restaurant / pub (20 to 100) | 8 | 100 (200) | 2 | 0.63 | 3,896 | 38.8 | 1.0 |
| 6 cinema / theatre (100 to 1000) | 20 | 1,000 (1,000) | 3 | 1.0 | 1,406 | 1.4 | 1.0 |
| 7 short-visit shop | 1,074 | 200 (6,000) | 16 | | 22 | 0.085 | 0.08 |
| 8 longer-visit shop / gym | 1,280 | 100 (10,000) | 13 | | 20 | 0.24 | 0.10 |
| 9 church (50 to 500) | 22 | 100 (1,000) | 2 | 0.95 | 1,430 | 14.3 | 1.0 |
| 12 hospital | 17 | 750 (750) | 2 | | 429 | 0.57 | 0 |
| 14 health centre | 95 | 1,000 (1,000) | 1 | | 32 | 0.03 | 0 |
| 22 nursing home | 8 | 55 (55) | 1 | | 113 | 2.0 | 1.0 |
| 33 classroom | 775 | 300 (979) | 38 | | 28 | 0.076 | 0 |

The decisive finding is in types 5, 6, 9 and 10. There are 8 restaurants, 20 cinemas and 22 churches in the whole file, and each is a candidate for every agent (about 57,000 agents per restaurant). These are aggregates of many real venues, so their capacity (100 for a restaurant) is the design capacity of one venue and their realised peak occupancy is 3,900 sampled agents. Fill fractions of 14 to 39 follow from using these capacities as they stand. Design capacity (12's B1) cannot be repaired by editing numbers: the number of real venues each file entry stands for is not in the data.

### 5.2 Recalculation rule actually used

Since B1 is not reconstructible offline, I used a peak-anchored rule (12's B2), stated explicitly and kept in `capacity_recalc.py`:

- residences: capacity = assigned members + 1 (12's fallback);
- every other type: capacity = 1.2 times the 95th percentile of the daily maximum occupancy in the reference run (minimum 2; the assigned count where the location was never occupied);
- area = capacity times the metres-squared-per-person figure of the type table (12 section 4.2), for the types that have one;
- type factors: effective removal per capacity person, (m3 per person) x (ACH + 1 per hour), divided by the home value (60), with low, mid and high ACH. All figures in the type table are the estimates in 11 and 12, not sourced values; I did not look anything up.

Validation (12 section 5), using a second 3-week reference run with a different seed as the held-out check (`summaries/p3_capacity_validation.json`):

| criterion | result |
|---|---|
| every location positive capacity and area | yes (types 1 and 15 left unchanged, no area for type 10) |
| 90% inside stated ranges for 5, 6, 9 | not applicable: capacities are now 4,632, 1,662 and 1,684 (aggregate scale), outside the per-venue ranges by construction |
| held-out peak occupancy above capacity in more than 1% of occupied locations | fails for types 7 (5.2%), 8 (3.6%), 13 (6.5%) and 3 (1.7%), passes for all other types |
| within-type distribution shape | max / median is 1 to 3 for types 5, 6, 9, 10, 12, 14, 22, 33, 44, and 12 to 70 for types 3, 4, 7, 8, 13 |
| totals against external aggregates | not done: no external data used. Residence total goes from 9.2 million to 514,000 |
| diff against old file | `p3_capacity_validation.json`: 72,529 of 73,036 residences, all or nearly all locations of every other type changed |

The peak-anchored rule is circular in one respect: it sets fill fraction near 0.8 at each location's daily peak by construction, so the volume model can only express diurnal and weekday fill variation, not venue-to-venue crowding differences. That is a limitation of what the data allow, not of the model.

## 6. Volume-model experiments (12 phase E)

All cells calibrated to early R 1.39 (two seeds, 4 weeks, tolerance 5%), 4 replicates of 8 weeks, multiplier off, no venue profile. "Household share" is the share of infections in residences (45.6% in the unchanged kernel). Fill fraction is occupancy at infection divided by the location's capacity in the file used for that cell.

| cell | what it tests | early R | early k | 95% CI | zero share | busiest 10% of venues | household share |
|---|---|---|---|---|---|---|---|
| unchanged kernel (phase 2) | baseline | 1.41 | 5.25 | 4.35 to 6.99 | 0.28 | 0.30 | 0.46 |
| E1: non-residential, peak-anchored, mid ACH | volume model on | 1.39 | 3.24 | 2.78 to 3.82 | 0.32 | 0.31 | 0.53 |
| E2: residences included | household denominator | 1.34 | 2.19 | 1.85 to 2.58 | 0.35 | 0.36 | 0.38 |
| E3a: low ACH | ventilation sensitivity | 1.37 | 3.23 | 2.74 to 3.96 | 0.32 | 0.35 | 0.45 |
| E3b: high ACH | ventilation sensitivity | 1.39 | 3.22 | 2.78 to 3.64 | 0.32 | 0.29 | 0.61 |
| E3c: fill pinned (head count times type factor, original capacities) | separates ventilation contrast from fill | 1.36 | 2.86 | 2.38 to 3.51 | 0.33 | 0.35 | 0.43 |
| E4: type-median capacity | removes within-type size variation | 1.37 | 1.96 | 1.66 to 2.33 | 0.34 | 0.44 | 0.27 |
| raw: original capacities as they stand | the capacity-error mechanism of 12 section 1 | 1.37 | 1.30 | 1.14 to 1.45 | 0.40 | 0.78 | 0.17 |
| E5a: events, 0.3% of social-venue days, 1000 times | explicit events at fixed volume | 1.85 | 0.076 | 0.063 to 0.102 | 0.39 | 0.48 | 0.46 |
| E5b: events, 1% of social-venue days, 200 times | explicit events at fixed volume | 1.42 | 0.48 | 0.36 to 0.64 | 0.38 | 0.41 | 0.47 |

Reading, in order of how well each is supported:

- **The 11 and 12 expectation holds.** A fixed-volume model moves early k from 5.2 to 3.2, inside the predicted 2 to 4 range and nowhere near 0.1.
- **The change comes from the per-type contrast, not from fill fraction.** The fill-pinned cell (which uses head count, so no fill-fraction information at all) gives 2.9 against 3.2 for the full volume model, with overlapping intervals. Low, mid and high ACH all give 3.2. So the volume model is, to within this precision, the existing venue profile in a different parameterisation: the "trap 1" of 11 is realised empirically. For comparison the covid venue profile at strength 1 gives 3.5.
- **Capacity error manufactures dispersion, as predicted in 12 section 1.** Using the original capacities as they stand brings early k to 1.3 and puts 78% of infections in the busiest 10% of venues. The mechanism is the one in the toy calculation (per-location multiplier errors, 14 to 39 fold here), and it produces an apparently strong dispersion reduction with no physics behind it. A reduction of this kind in any future volume-model run needs the capacity audit before it is believed.
- **Within-type size variation is a modest contributor.** Replacing it by the type median (E4) lowers k from 3.2 to 2.0, but 46% of locations then exceed capacity, so the cell mixes size error with real variation.
- **Including residences** (E2) lowers k further to 2.2 and household share to 38%. Which is closer to the truth depends on dwelling data that the file does not contain (12 section 4.3).
- **Explicit events are the only route below 1 without the multiplier.** Events on 0.3% of social-venue days at 1000 times reach early k of 0.076 (0.063 to 0.10), in the published range, and 1% at 200 times reaches 0.48. Two cautions: (i) the early R in E5a, pooled over four replicates, is 1.85 against the 1.39 target, because a few huge events dominate; the two-seed calibration did not match it. A cell with the same k at matched R has not been run. (ii) The bootstrap intervals resample infectors, but infectors from one event are correlated, so the real intervals are wider. Events here act on every visitor of an aggregate venue for a whole day, which is a coarse stand-in for a single gathering.

Hold-out (E6): the E1 volume model refits the wave-1 area (8,131 against 8,565) and fails the wave-2 hold-out (section 3).

Levers (E7), uncontrolled early R 1.3 (volume model, non-residential, peak-anchored, mid ACH), reduction in infections over 12 weeks:

| lever | reduction |
|---|---|
| masks | 66.1% |
| isolate diagnosed + household | 40.9% |
| close venue types 5, 6, 8, 9 | 26.8% |
| ventilation of types 5, 6, 8, 9 (hazard halved) | 20.2% |
| isolate diagnosed only | 9.0% |

The volume-model arm is in the Set I regime (household isolation beats venue closure), not in the Set II one. Ventilation of the social venues (halving their hazard) achieves about three quarters of what closing them achieves (20% against 27%), as expected for a halved hazard against a removed one. The occupancy cap was not run (section 1).

## 7. Corrections and consequences for earlier notes

1. 09 section 6: the statement that "only the multiplier" produces chains over 50 cases is withdrawn (section 2 here). The flat shape-0.04 arm has 17 of 38,630.
2. 09 section 4 and 10 section 3: the lever reversal is confirmed at strengths 2, 4 and 8, but it is a statement about where the hazard sits, not about k (section 4 here).
3. 10 section 5: shape 0.3 is confirmed as the only corner not rejected by the wave-2 hold-out, now with 12 replicates against an 8-replicate baseline. The volume model joins the corners that fail.
4. 11 section 3 and 12 section 6: the predicted outcome (k from 5 to the 2 to 4 range) is what was observed, and the capacity-error warning of 12 section 1 is confirmed empirically.

## 8. What I could not do, and the open questions

- Design capacity (B1) and the external aggregate checks need Hungarian dwelling, class-size and bed data that I did not use. The peak-anchored rule is a stand-in, not a recommendation.
- The occupancy cap lever was not implemented.
- E5 is not matched on early R, and has no calibrated rerun.
- The ACH and removal figures are the estimates in 11 and 12 and were not checked against any standard.
- 12 section 7, question 3 (are hospitals subdivided into wards?): they are not. 17 hospital locations serve the whole file, each with about 430 occupants at peak, so the well-mixed assumption is applied to a building of that size. This is the setting in which the caregiving tail in section 2 arises.
- Replicate counts remain low (4 per cell outside the chain and shape-0.3 experiments), and calibration targets are known only to about 15% for the fitted waves.

## 9. How to rerun

```
cd superspreader/study
PANSIM_BIN=<old binary> python3 run_phase3.py levers waves tail    # phase-2 binary
python3 run_phase3b.py reference reference2
python3 capacity_audit.py ../../inputRealExample/locations0.json results/p3/reference_uncontrolled/occupancy.tsv summaries/p3_capacity_audit_original
python3 capacity_recalc.py ../../inputRealExample/locations0.json summaries/p3_capacity_audit_original_per_location.csv results/p3/inputs /tmp/audit_b_per_location.csv
python3 run_phase3b.py volume vollevers volwaves
python3 analyze_phase2.py chains lever waves
python3 analyze_phase3.py vol vol_ kernel_alpha1
python3 chain_structure.py
```

`PANSIM_BIN` selects the binary; the phase-2 jobs in this note ran with the binary built before the new flags, and all flags default to off with bit-identical output (checked on one 8-week run).
