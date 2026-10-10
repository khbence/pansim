# ACT-049 phase 2: review of 07 and 08, and the runs they asked for

10 October 2026. This note answers `07_review_of_06_and_next_steps.md` and executes `08_next_phase_run_plan.md` (both tracks). Everything here was run with the code in `superspreader/study/` (`run_phase2.py`, `analyze_phase2.py`, `early_window.py`, `occupancy_check.py`) and the new `--densityExponent` flag. Summaries are in `study/summaries/p2_*.json` and `early_*.json`. Raw traces stay local.

## 1. Review of 07 and 08

**What I agree with, and checked.**

- The kernel algebra in 07 section 2 is right. In `include/policies/infectionPolicies.h` the hazard is `1 - exp(-k * (sum infectiousness present / people present) * locationInfectiousness * eventBoost * seasonality * dt)`, so one infector's expected offspring in a venue is occupancy-independent, and offspring variance at a venue is at most Poisson. Occupancy and venue multipliers therefore concentrate infections in places, not in infectors. Overdispersion in this kernel can only come from per-agent variation: natural history, time allocation, or a per-infection multiplier.
- The attribution sentence in 06 Experiment 1 ("the mild tail is the occupancy distribution, not who shares a household") is not supported. The shuffle changed k from 4.56 to 4.39, which says only that household membership is not the source. The `--densityExponent` runs below test the occupancy claim directly.
- 06 Experiment 2 was not matched on early growth (early R 0.95 versus 3.36, early k 0.86 versus 3.06). "Not supported" should be "not tested".
- 06 Experiment 4 compared arms with different mean offspring. Redone with arms matched on mean offspring, the zero-share result reverses (section 5).

**One thing I found that 07 did not.** The whole-window k used in 06 for the venue strength 4 and 8 cells is contaminated by depletion, because those epidemics infect nearly the whole simulated population within eight weeks. Measured only on infectors infected in days 1 to 10 with offspring counted over the next 14 days, the figures are:

| venue strength | whole-window k (06) | early-window k |
|---|---|---|
| 4 | 0.85 | 3.1 |
| 8 | 0.54 | 1.1 |

Venue amplification does not take early k below 1 even at strength 8 (1.1). The sub-1 values in 06 were mostly saturation. The same early-window protocol shows the multiplier cells hold up: early k is close to the gamma shape in every cell (section 3).

**Changes I made to 08.**

- Every comparison across cells is calibrated to the same early reproduction number (target 1.39, the baseline value at days 1 to 10) with a deterministic log-log secant on the infection coefficient, accepting within 5% (seeds fixed, resumable).
- Index cases are infected on days 1 to 10 with a known infector; the initial `-r` seeds are excluded because they start part-way through infectiousness. Offspring are counted within 14 days; the k interval is a bootstrap over infectors.
- `--meanSchedules` and `--fixedInfectiousness` were not implemented. Schedules are discrete per agent type and the progression engine is stochastic, so there is no clean definition of a "mean schedule" or a fixed infectiousness that I would trust. The `--densityExponent` flag and the shuffle together answer the same question.
- A2 "halve the venues" is covered by the density exponent comparison plus the new `occupancy` and `infectiousPresent` trace columns (occupancy at the moment of infection).
- The quanta variant in A4 is scope only, not run.

## 2. Kernel and occupancy (track A1, `--densityExponent`)

The exponent α in density = (sum infectiousness present) / (people present)^α. α = 1 is the original kernel. α = 0 removes the dilution, so expected offspring per venue grows with occupancy. All three cells are calibrated to early R 1.39, with the shed multiplier off and the venue profile off. Four replicates, eight weeks.

| α | measured early R (target 1.39) | k (early window) | 95% CI | zero share | top 20% of infectors' share | median occupancy at infection | infections in busiest 10% of venues |
|---|---|---|---|---|---|---|---|
| 1 (original) | 1.378 to 1.411 | 5.25 | 4.35 to 6.99 | 0.28 | 0.50 | 11.5 | 0.30 |
| 0.5 | 1.386 | 2.66 | 2.27 to 3.19 | 0.33 | 0.53 | 649 | 0.51 |
| 0 | 1.396 | 1.79 | 1.58 to 2.04 | 0.37 | 0.57 | 1235 | 0.58 |

Reading:

- The original kernel gives early k of about 5, a little higher than the whole-window 4.5 in 06. Real schedules plus the original kernel are a mild tail.
- Making transmission scale with crowd size does lower k, from 5.2 to 1.8, and moves most infections into the largest venues (median occupancy at infection 12 to 1235). Even with no dilution at all, k stays above 1.5. Occupancy alone cannot reach the published range of 0.1 to 0.5.
- This confirms 07's point from the other direction: under the present kernel, occupancy cannot be what produces k of 5, because expected offspring does not depend on it. By elimination, since the kernel allows nothing else, the remaining dispersion at α = 1 comes from time allocation and natural history; I did not decompose it further.
- `occupancy_check.py` (infections per assigned agent by venue size quintile) was inconclusive, because assigned size is not presence. The trace columns above supersede it.

## 3. Heterogeneity grid (track A2)

Gamma per-infection multiplier (shape: off, 1, 0.3, 0.1) crossed with the covid venue profile (off, strength 1). Each cell is calibrated to early R 1.39 on two seeds and has 4 replicates; the four-replicate early R differs from the target by up to 9% (the shape 0.3 plus covid cell is highest at 1.52). Shuffled schedules are off.

| multiplier shape | venue profile | early R | early k | 95% CI | zero share | top 20% share | CV² |
|---|---|---|---|---|---|---|---|
| off | off | 1.41 | 5.25 | 4.35 to 6.99 | 0.28 | 0.50 | 0.90 |
| off | covid | 1.48 | 3.51 | 3.03 to 4.24 | 0.29 | 0.51 | 0.96 |
| 1 | off | 1.37 | 0.88 | 0.77 to 0.99 | 0.44 | 0.64 | 1.86 |
| 1 | covid | 1.45 | 0.69 | 0.60 to 0.82 | 0.43 | 0.65 | 2.13 |
| 0.3 | off | 1.34 | 0.33 | 0.29 to 0.37 | 0.59 | 0.79 | 3.76 |
| 0.3 | covid | 1.52 | 0.30 | 0.27 to 0.33 | 0.58 | 0.80 | 4.00 |
| 0.1 | off | 1.35 | 0.127 | 0.108 to 0.148 | 0.73 | 0.95 | 8.63 |
| 0.1 | covid | 1.35 | 0.119 | 0.106 to 0.138 | 0.75 | 0.96 | 9.17 |

- Early k tracks the shape. The gamma composition rule, 1/k_total = 1/k_base + 1/shape + 1/(k_base * shape), predicts 0.72, 0.24 and 0.083 for shapes 1, 0.3 and 0.1 with a base k of 5.25. The measured values (0.88, 0.33, 0.127) are 20% to 50% above that, so the base dispersion adds less than the composition rule assumes. Early k is within a factor of 1.3 of the shape in every multiplier cell.
- The covid venue profile at strength 1 lowers k only modestly (5.25 to 3.51 with no multiplier, 0.127 to 0.119 with shape 0.1). It is a minor contributor next to the multiplier, in agreement with 06 and with the kernel algebra.
- A multiplier of shape 0.1 gives the published "about 10% of infectors cause 80% of infections" (top-20% share 0.95 to 0.96). It is the only mechanism in the study that gets there.

## 4. Same-k intervention comparison, rebuilt (track A3, lever experiments)

06 Experiment 2 compared arms that differed in early growth. The rebuild uses two arms, each calibrated so the early R is about 1.3 and each run from the same seeds so the four replicates are paired:

- Set I, "biology": shape-1.0 gamma multiplier, venue profile off (a multiplier of shape 1 gives early k of about 0.9 in the grid).
- Set II, "setting": multiplier off, covid venue profile at strength 8 (strength 8 alone gave early k of about 1.1 in the uncontrolled sweep).

Levers start on day 21, with no intervention before then, and run for 12 weeks (`levers/*.json`): masks (non-home infectiousness coefficient 0.6); closing restaurants, large evening venues, long-visit venues and churches (types 5, 6, 8, 9); isolation of diagnosed agents only (quarantine policy 1); isolation of diagnosed agents and their household (policy 2). Measured early R: 1.30 for Set I and 1.17 for Set II. Set II landed below its target, so the arms are matched to within 10% rather than 5%; the relative comparisons below are within each arm.

Infections over the 12 weeks, relative to no intervention (paired, mean over four replicates, replicate range in brackets):

| lever | Set I reduction | Set II reduction | Set I quarantine person-days | Set II quarantine person-days |
|---|---|---|---|---|
| none | 54,784 infections | 20,928 infections | 0 | 0 |
| masks | 73.6% (72 to 77) | 78.8% (78 to 80) | 0 | 0 |
| close venue types | 37.2% (33 to 40) | 75.3% (73 to 78) | 0 | 0 |
| isolate diagnosed + household | 53.3% (51 to 56) | 34.2% (31 to 38) | 269,335 | 165,324 |
| isolate diagnosed only | 16.9% (15 to 21) | 23.4% (21 to 25) | 119,348 | 48,527 |

Peak hospital occupancy (simulation agents; multiply by about 55.8 for the national scale), mean over four replicates:

| lever | Set I | Set II |
|---|---|---|
| none | 547 | 231 |
| masks | 106 | 60 |
| close venue types | 273 | 60 |
| isolate + household | 202 | 142 |
| isolate only | 423 | 173 |

What this says:

- The ranking depends on where the heterogeneity sits. When it is a setting property (Set II), closing venue types is as good as masks (75% versus 79%). When it is a per-person property (Set I), closing venues is half as good (37% versus 74%) because the offspring that matter are spread across homes and schools.
- Household isolation is more effective under person-level heterogeneity (53% versus 34%). A plausible reason is that removing a diagnosed person and their household takes out the people who carry the transmission, whereas in Set II the transmission happens in the venues; I did not test that explanation.
- Per 1000 quarantine person-days, household isolation averts about 108 infections in Set I and 43 in Set II; agent-only isolation averts about 77 and 101. These are single-configuration, four-replicate estimates and I would not read the Set II ordering as firm.
- Agent-only isolation is the weakest lever in both arms (17% and 23%). It acts only on diagnosed cases.
- The decision-relevant split is where the heterogeneity lives, not the value of k. Neither arm has a k near 0.1.

## 5. Ascertainment audit at matched R (track A3, audit)

Early R matched at 2.5 (6-week runs, 3 replicates), comparing a flat arm (no multiplier) with a shape-0.1 multiplier arm. Per-replicate k from each sampling scheme:

| scheme | flat arm: mean offspring, zero share | flat arm: k by replicate | shape 0.1 arm: mean offspring, zero share | shape 0.1 arm: k by replicate |
|---|---|---|---|---|
| true | 0.57, 0.58 | 1.9, 61, 11.5 | 0.69, 0.80 | 0.137, 0.135, 0.120 |
| symptom-forward | 0.22, 0.79 | infinity (all 3) | 0.29, 0.87 | 0.97, 0.12, 0.115 |
| mixed backward | 0.69, 0.45 | infinity (all 3) | 0.96, 0.65 | 0.49, 0.27, 0.26 |
| pure backward (true offspring of the sampled sources) | 1.53, 0.00 | infinity | 4.51, 0.00 | 2.3, 1.8, 1.4 |
| cluster | 0.48, 0.63 | infinity, infinity, 3.7 | 0.51, 0.84 | 0.143, 0.166, 0.086 |

- With true offspring, the shape-0.1 arm recovers k of 0.12 to 0.14. Forward symptom sampling misses most of it (one replicate gives 0.97 against truth 0.14); backward tracing of true offspring inflates the apparent mean offspring by a factor of 6.5 (0.69 to 4.5) and k to 1.4 to 2.3.
- In the flat arm every non-true scheme returns infinity, because the sampled offspring variance is below the mean. The true-offspring k in that arm is 1.9, 61 and 11.5 across replicates, so it is noisy at about 250 infectors per replicate and I would not interpret it beyond "well above 1".
- In the shape-0.1 arm, mixed backward sampling raises k from 0.13 to 0.26 to 0.49 (2 to 4 times) and pure backward sampling to 1.4 to 2.3 (10 to 17 times). Detection and tracing choices can therefore move the observed k by a factor of up to about 17, but a dispersion near 0.1 is not turned into one near 5 by sampling alone.

## 6. Subcritical chains with matched mean offspring (track B, experiment 4)

About 1000 seeds per run (`ExposeToMutation` with parameter 0.0054, about 1000 index cases), 7 weeks, 4 replicates each, so each cell has about 3800 independent chains. Each arm is calibrated to an all-node mean offspring of 0.33 (this is the comparison 07 asked for; 06 compared arms with different means). Shed multipliers use gamma shapes 0.125, 0.04 and 0.09, taken from the 08 table so that arms target k of about 0.13, 0.05 and 0.09. "Isolate" is quarantine policy 1.

| arm | mean offspring (all nodes) | all-node k | zero share (95% CI) | mean chain size | P(chain > 10) | chains > 50 | max chain |
|---|---|---|---|---|---|---|---|
| flat, shape off | 0.315 | 5.1 | 0.737 (0.726 to 0.749) | 1.46 | 0.0003 | 0 | 11 |
| flat, isolate | 0.307 | 6.5 | 0.741 (0.729 to 0.752) | 1.44 | 0.0005 | 0 | 13 |
| caregiving profile, shape off | 0.345 | 4.3 | 0.718 (0.706 to 0.729) | 1.52 | 0.0010 | 0 | 17 |
| flat, shape 0.125 | 0.309 | 0.134 | 0.856 (0.846 to 0.865) | 1.45 | 0.0092 | 0 | 25 |
| flat, shape 0.04 | 0.316 | 0.054 | 0.909 (0.901 to 0.916) | 1.46 | 0.0134 | 0 | 45 |
| flat, shape 0.04, isolate | 0.255 | 0.048 | 0.915 (0.907 to 0.922) | 1.34 | 0.0068 | 0 | 42 |
| caregiving + shape 0.09 | 0.321 | 0.088 | 0.860 (0.851 to 0.869) | 1.47 | 0.0058 | 3 | 116 |
| caregiving + shape 0.09, isolate | 0.299 | 0.100 | 0.866 (0.857 to 0.875) | 1.43 | 0.0039 | 2 | 122 |

What this says:

- 06 concluded that heterogeneity did not raise the zero share. That was a mean-offspring confound. At matched mean offspring (0.31 to 0.35), the zero share moves from 0.72 to 0.74 in the flat arms up to 0.86 to 0.91 with shape 0.125 to 0.04. The 95% intervals do not overlap. The 92% zero share associated with a k of 0.036 is nearly reached: the shape 0.04 arm gives 0.909 with an all-node k of 0.054.
- Heterogeneity does not change the mean chain size (1.43 to 1.52, since mean offspring is fixed) but does thicken the tail: P(chain > 10) rises from 0.0003 (flat) to 0.0134 (shape 0.04), and chains longer than 50 appear only in the caregiving plus shape-0.09 arm (3 of 3808, 95% interval for P(>50) 0.0003 to 0.0023).
- The 50-case chain that 06 found absent is not a sample-size question. Flat arms (3800 chains each) and the plain caregiving arm have none. Without a multiplier the kernel's offspring variance is at most Poisson at each venue, so a mean of 0.33 cannot produce a long chain. Only the per-agent multiplier does.
- Isolating diagnosed cases (same infection coefficient as the partner arm, so the mean offspring is not re-matched) lowers mean offspring by 3% to 19% and the probability of a chain over 10 by a third to a half in the shape arms (0.0134 to 0.0068; 0.0058 to 0.0039). It does not remove the tail. The differences between isolate and non-isolate in the flat and plain caregiving arms are within sampling noise.
- The flat arm's all-node k of 5 to 6 at R 0.3 is the estimator's behaviour at a low mean: the zero share of 0.74 is roughly Poisson (e^-0.31 = 0.73).

## 7. The fitted first two waves, recalibrated (track A5)

Each heterogeneity corner is refit so that the first 105 days' hospital area matches the baseline wave-1 area (target 8,565), then wave 2 is a hold-out. Four replicates per corner, 37 weeks from 23 Sep 2020, with the fitted closures and quarantine policy 3. Peak hospital occupancy is in simulation agents (the reconstruction target is about 144 for wave 1 and about 225 for wave 2). Mean (SD) over replicates:

| corner | wave 1 peak | wave 2 peak | wave 1 area (105 d) | total infections | early k |
|---|---|---|---|---|---|
| baseline (4 replicates, 2 original plus 2 new seeds) | 167 (23) | 222 (18) | 9,895 (1,560) | 52,509 | 3.53 (2.66 to 4.82), 2 new seeds only |
| multiplier shape 0.3 | 142 (21) | 252 (57) | 8,232 (1,220) | 51,817 | 0.32 (0.28 to 0.37) |
| multiplier shape 0.1 | 132 (41) | 152 (36) | 7,259 (1,657) | 39,504 | 0.092 (0.070 to 0.130) |
| covid venue profile, strength 1 | 156 (33) | 148 (18) | 8,188 (1,363) | 39,355 | 3.46 (2.79 to 4.54) |
| venue strength 4 | 177 (39) | 120 (29) | 8,318 (1,567) | 34,062 | 1.99 (1.60 to 2.66) |
| shape 0.3 plus venue strength 1 | 154 (15) | 105 (33) | 8,530 (646) | 33,068 | 0.26 (0.23 to 0.30) |

Notes:

- The first calibration of venue strength 4 failed: the epidemic at this strength is threshold-like in the infection coefficient, and the secant oscillated between 200 and 260,000. I bracketed it by hand (coefficient 1.2e-4, which gave a wave-1 area of 8,318, about 3% below target). The first-pass shape 0.1 cell stopped at 6,410 on its two calibration seeds, and its four-replicate area is 7,259, 15% below the target. The calibration seeds and the replicate seeds are the same two seeds plus two more; the replicate-to-replicate SD of the area is about 1,200 to 1,700 (15% of the mean) in every cell, so the 8,565 target is itself only known to about that level. The baseline replicate set averages 9,895.
- Wave 1 hospital area can be refit by changing the infection coefficient in every corner (the areas match the baseline to within about 15%, which is the replicate noise). The wave-2 hold-out is where they differ. Four of five corners undershoot the baseline wave 2 (222): shape 0.1 at 152, venue strength 1 at 148, venue strength 4 at 120, and the stack at 105, a 32% to 53% shortfall that is several replicate SDs. Shape 0.3 overshoots at 252 with a wide range (198 to 329).
- Total infections fall by 25% to 37% (from 52,509 to between 33,068 and 39,504) in the shape 0.1, venue strength 1, venue strength 4 and stack corners, for the same wave-1 hospital area. Shape 0.3 is within 1% of the baseline.
- A model with dispersion near 0.1 to 0.3 can be recalibrated onto the wave-1 hospital area, but wave 2 then changes (down for most corners, up for shape 0.3). That bears directly on whether the multiplier can be dropped into the fitted model: it cannot be added with only the infection coefficient refit.

## 8. Corrections to 06

1. Experiment 1 attribution: the sentence "the mild tail is the occupancy distribution, not who shares a household" should read "the shuffle leaves k unchanged, so household membership is not the source. The occupancy-density runs show occupancy cannot explain it either; the tail comes from time allocation and natural history."
2. The venue strength 4 and 8 whole-window k (0.85, 0.54) are depletion artefacts. Early-window k is 3.1 and 1.1. The statements that "stronger amplification can cross k = 1" should be changed to say that strength 8 reaches 1.1 at best.
3. Experiment 2: "not supported" should be "not tested". The arms were not matched on early growth. Section 4 here is the replacement.
4. Experiment 4: heterogeneity does raise the zero share once the mean offspring is matched, and the shape 0.04 multiplier reaches 0.91 against 0.74 for flat. The absence of a 50-case chain in the flat arms is a kernel dispersion limit, not a sample-size one.

## 9. What is still open

- k is not identified by anything in this study from the fitted epidemic. No hospital curve, in any corner, distinguishes k of 0.1 from k of 5 on wave 1. Wave 2 hints at it, but with four replicates and a calibration target that is only known to 15%.
- The multiplier and the venue profile are not independent of the progression engine's own heterogeneity; I did not decompose the base k of 5 into its natural-history and time-allocation parts.
- The set II quarantine and venue-closure estimates are for one configuration of venue strength 8 and a 12-week window. I would repeat them at two other strengths before using them in a decision.
- Replicates are 4 per cell. Intervals on k are bootstrap intervals over infectors within a pooled sample, so they are narrower than a replicate-level interval would be.

## 10. How to rerun

```
cd superspreader/study
python3 run_phase2.py kernel chains grid audit lever waves
python3 analyze_phase2.py kernel chains grid audit lever waves
```

Calibration state is in `results/p2_calibration_log.json`; finished jobs are skipped on rerun. The venue strength 4 wave cell was bracketed by hand at coefficient 1.2e-4 (4 replicates in `results/p2/waves_venue4cal`); the failed automatic run is in `waves_venue4` and is excluded from the table.
