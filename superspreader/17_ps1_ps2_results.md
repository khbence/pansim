# ACT-049 PS1 and PS2

11 October 2026. Persistent individual traits against the per-episode multiplier, and the time profile of infectiousness. Instructions: the ACT-049 note `16_pansim_agent_instructions.md`. Code: `superspreader/study/ps/`. Summaries: `study/results/ps/ps1/summary.json` and `study/results/ps/ps2/`.

## Deviations

The full PS1 grid was run (10 arms, 12 replicates). Nothing in PS1 was dropped for time. A 4-week job is about 15 seconds and a 10-week job about 31 seconds on this GPU, so the grid was a few hours, which is inside the budget in section 2 of the instructions.

- **T5 was skipped.** There is no single well-mixed location mode. Building one would have meant a second kernel, so the mean-field check stays the attached toy, not a PanSim run.
- **Susceptibility is applied on the hazard scale.** The weight `u` enters as `1 - (1 - p)^u`, where `p` is the existing infection probability, including whatever susceptibility age, variant and vaccination have already set. It is not a replacement, and it is not a raw multiplication of `p`, which would exceed 1 for large `u`.
- **The infection coefficient falls as coupling rises**, where the instructions expected it to rise. The early-R window is the size-biased cohort (days 1 to 10, seeds excluded), so coupling inflates measured early R at a fixed coefficient and the secant lowers the coefficient. The two-seed calibrations all landed inside 5% of 1.39.
- **P100ln's twelve-replicate early R is 1.54**, against a two-seed calibration of 1.45 (inside the 5% band around 1.39). The comparison of its final size with the gamma arm is not at matched early R.
- **The 50% peak window is days 1 to 19** (`f_peak` 0.95). The profile in this natural history is a broad hump, not a short spike, so peak mode at that definition is close to the constant multiplier. This is a result, recorded in the PS2 section, not a substitution.
- **No `figure-style` skill is on this machine.** Figures use one local style function, applied before plotting. Titles are claims the script asserts.
- **T6 does not fully pass.** The multiplier itself has mean 1.006 and CV² 3.74 against 1/shape = 3.70. The mean infectiousness profile of the peak-window run is 4.2% below the constant-multiplier run, outside the 2% bar, and days 20 and 21 (past the window, 60 to 90 agents) differ by more than 15%. The implied episode-total CV² is within 15% of `f_peak² / shape`.

## Answer

**Persistence without coupling changes nothing.** At matched early R, the per-episode gamma of shape 0.3 (arm E) and the same gamma stored for life and applied only to infectiousness (arm P0) give early k 0.311 and 0.295 (ratio 0.95) and final sizes 48,274 and 50,684 (ratio 1.05). Cohort k intervals overlap in every window. H1 passes on the pilot and again on all 12 replicates.

**Coupling is what changes the epidemic.** Early k rises from 0.30 at no coupling to 0.47, 0.63 and 0.95 at rho 0.25, 0.5 and 1, and final size falls from 50,684 to 38,695, 21,808 and 12,680. H4 passes. The pre-registered band, composition k lifted 20 to 50%, holds for the uncoupled arms and fails for every coupled arm: those sit within about 10% of the composition value itself, below the band. H2 fails. A lognormal trait at full coupling gives early k 0.46, half the gamma value. H6 passes.

**Cohort k does not rise as high-trait people are depleted.** The ratio of k in days 41 to 50 over k in days 1 to 10 has a replicate interval that includes 1 in every main arm. H5 passes. The mean trait of people infected does sit above 1 when coupling is on, and at full coupling it falls across the five windows by a margin the replicate intervals separate.

**Shape 0.05 does not reach k = 0.33 once coupling is on.** At rho 0.5 the early k is 0.46; at rho 1 it is 0.83. That is the documentation the instructions asked for, and it was not iterated further.

**A peak-window multiplier does not change which lever wins.** The natural-history profile is a broad hump peaking on days 6 and 7, and half-maximum covers days 1 to 19 (95% of the area). Household isolation cuts infections by about half, and isolating only the diagnosed person cuts them by about a sixth, under both the constant multiplier and the peak window. The paired differences include zero. Only 23% of infectiousness falls before the median diagnosis, so the prediction that a peak window would weaken isolation was expected to fail, and it did.

## Instrumentation

New flags, default off. With every new flag left at its default, an 8-week run matches the previous binary on the daily table and on the first 13 trace columns (T1, seeds and coefficient of the phase 2 cells). `--sheddingShape` and `--traitShape` together are an error. The trait is drawn from its own generator, seeded by the run seed and the agent id, so the same seed gives the same trait vector in every arm (T3) and other random draws are not shifted.

| flag | behaviour |
|---|---|
| `--traitShape`, `--traitDist`, `--traitPersist` | gamma or lognormal, mean 1, CV² = 1/shape, drawn once at the start and kept |
| `--traitSusceptibilityExp` | weight `trait^rho`, scaled to mean 1, applied on the hazard scale |
| `--sheddingMode peak` and `--sheddingPeakWindow t1,t2` | the per-episode multiplier acts only on those days since infection; outside it, the multiplier is 1 |
| `--traitDump`, `--profileOut` | per-agent trait file; mean infectiousness by day since infection |

The trace gained the infector's multiplier and infection time, symptom onset, and the quarantine interval. Symptom onset is the first day the well/ill state leaves "well".

Tests, all reported:

| test | result |
|---|---|
| T1 | pass. Daily table and the original trace columns are identical for no multiplier and for shape 0.3 |
| T2 | pass. Gamma shape 0.3: mean 1.003, CV² 3.34 against 3.33. Shape 0.08: mean 0.997, CV² 12.62 against 12.5. Lognormal shape 0.3: mean 1.003, CV² 3.22 against 3.33. Susceptibility mean is 1 within 1% in each case |
| T3 | pass. Seed 5 gives the same trait vector at rho 0, 0.5 and 1. Seed 6 differs |
| T4 | pass. In an 8-week run, 62 agents were reinfected and every infected agent's recorded multiplier equalled the trait drawn at creation |
| T5 | skipped, as above |
| Regression | pass. Same seeds and coefficients as 09: early k 5.25 (published 5.25, interval 4.35 to 6.99) and 0.332 (published 0.33, interval 0.29 to 0.37) |

Reinfection exists. Recovered agents return to susceptible after about three weeks, and a persistent trait is the same trait on the next episode. There is already per-agent susceptibility from vaccination and prior infection; the new weight multiplies the infection probability on top of it.

## Design

Szeged, the existing agent and location files, venue model off, venue profile off, density exponent 1, no events, real schedules, no closures, quarantine policy 0, `-r` as in phase 2. Ten weeks, so an infection on day 50 still has 14 days of follow-up. Calibration seeds 9001 and 9002. Replicate seeds 1101 to 1112. None of these were used in phases 1 to 3.

The infection coefficient is a log-log secant to early R 1.39 within 5%, two seeds, four weeks, with the saturation guard from phase 2. Early k uses the unchanged protocol: index cases infected on days 1 to 10 with a known infector, offspring within 14 days, method of moments. Intervals are a cluster bootstrap over the 12 replicates (2,000 resamples) and, separately, a t interval of the 12 replicate-level estimates. The infector-level bootstrap from 09 is narrower and is not the one used for the verdicts below.

P25k solved the shape at rho 0.25 so that early k on the calibration seeds was within 15% of arm E. One pass was enough: shape 0.08 gave early k 0.339 against E's pilot value 0.314. The twelve-replicate early k is 0.270, which is 13% below E's twelve-replicate value 0.311, still inside 15%.

## PS1 table

Twelve replicates. "Cluster" is the bootstrap over replicates. "Replicate t" is the t interval of the twelve replicate estimates. Final size is ever-infected at day 70.

| arm | coefficient | early R | early k | cluster interval | replicate t | zero share | top 20% | final size | infectors |
|---|---:|---:|---:|---|---|---:|---:|---|---:|
| A0 | 4.10e-4 | 1.383 | 4.66 | 4.09 to 5.47 | 4.09 to 5.77 | 0.30 | 0.50 | 59,574 | 12,566 |
| E | 4.84e-4 | 1.333 | 0.311 | 0.288 to 0.336 | 0.289 to 0.347 | 0.59 | 0.80 | 48,274 | 11,201 |
| P0 | 4.84e-4 | 1.331 | 0.295 | 0.267 to 0.325 | 0.272 to 0.341 | 0.59 | 0.80 | 50,684 | 12,214 |
| P25 | 2.89e-4 | 1.366 | 0.471 | 0.427 to 0.520 | 0.440 to 0.555 | 0.51 | 0.72 | 38,695 | 8,163 |
| P50 | 1.99e-4 | 1.336 | 0.632 | 0.556 to 0.720 | 0.597 to 0.770 | 0.47 | 0.68 | 21,808 | 5,338 |
| P100 | 1.37e-4 | 1.337 | 0.954 | 0.854 to 1.090 | 0.879 to 1.163 | 0.42 | 0.63 | 12,680 | 3,603 |
| P25k, shape 0.08 | 1.59e-4 | 1.377 | 0.270 | 0.240 to 0.308 | 0.254 to 0.321 | 0.60 | 0.82 | 17,855 | 3,751 |
| P100ln | 1.84e-4 | 1.541 | 0.464 | 0.426 to 0.511 | 0.430 to 0.534 | 0.44 | 0.69 | 26,050 | 6,308 |
| P50, shape 0.05 | 7.09e-5 | 1.313 | 0.455 | 0.403 to 0.536 | 0.441 to 0.576 | 0.53 | 0.73 | 6,984 | 1,516 |
| P100, shape 0.05 | 4.54e-5 | 1.288 | 0.834 | 0.665 to 1.028 | 0.748 to 1.087 | 0.45 | 0.65 | 4,220 | 924 |

Cohort k, replicate mean, and the ratio of the last window to the first. The ratio interval is a t interval across the 12 replicates.

| arm | days 1-10 | 11-20 | 21-30 | 31-40 | 41-50 | ratio 5/1 | interval |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 | 4.93 | 5.00 | 5.29 | 5.26 | 5.47 | 1.17 | 1.00 to 1.35 |
| E | 0.318 | 0.318 | 0.317 | 0.319 | 0.333 | 1.07 | 0.97 to 1.17 |
| P0 | 0.307 | 0.313 | 0.316 | 0.312 | 0.329 | 1.11 | 0.96 to 1.25 |
| P25 | 0.498 | 0.500 | 0.496 | 0.498 | 0.490 | 1.01 | 0.90 to 1.13 |
| P50 | 0.683 | 0.682 | 0.684 | 0.684 | 0.674 | 1.03 | 0.85 to 1.21 |
| P100 | 1.021 | 0.968 | 0.971 | 1.025 | 0.950 | 0.97 | 0.84 to 1.10 |
| P100ln | 0.482 | 0.496 | 0.449 | 0.480 | 0.490 | 1.06 | 0.86 to 1.26 |
| P25k | 0.287 | 0.300 | 0.282 | 0.285 | 0.289 | 1.05 | 0.89 to 1.20 |

Mean trait of the infected cohort, same windows. The formula value is `(a + rho) / a` for the first window.

| arm | formula | 1-10 | 11-20 | 21-30 | 31-40 | 41-50 | first vs last intervals separate |
|---|---:|---:|---:|---:|---:|---:|---|
| E | 1.00 | 1.00 | 1.01 | 1.01 | 1.00 | 1.00 | no |
| P0 | 1.00 | 0.99 | 1.00 | 1.01 | 1.01 | 1.00 | no |
| P25 | 1.83 | 1.64 | 1.64 | 1.65 | 1.61 | 1.59 | no |
| P50 | 2.67 | 2.31 | 2.30 | 2.32 | 2.23 | 2.22 | no |
| P100 | 4.33 | 3.67 | 3.61 | 3.62 | 3.45 | 3.42 | yes |
| P100ln | toy about 3.4 | 3.08 | 2.97 | 2.86 | 2.55 | 2.34 | yes |
| P25k | 4.13 | 3.44 | 3.42 | 3.37 | 3.36 | 3.30 | no |

Rank correlation between trait and infection day, among infected agents with a known infector, is about 0 for E and P0 and between −0.03 and −0.08 for the coupled arms. Negative means higher trait earlier, which is the size-bias direction, and the correlations are small because infection day also depends on who is nearby.

## Pre-registered predictions

| id | verdict | numbers |
|---|---|---|
| H1 | pass | k ratio P0/E = 0.95, inside 0.75 to 1.33. Final-size ratio 1.05, inside 0.90 to 1.10. Cohort k intervals overlap in all five windows |
| H2 | fail | E and P0 are inside the 20 to 50% band (0.311 and 0.295 against 0.29 to 0.36). P25, P50 and P100 are below it: 0.47 against 0.51 to 0.64, 0.63 against 0.71 to 0.89, 0.95 against 1.08 to 1.36. Early k does rise with coupling |
| H3 | mixed | E and P0 are flat at 1. Coupled arms are above 1 but 10 to 15% below `(a+rho)/a`. A decline that the replicate intervals separate is present at rho = 1 only |
| H4 | pass | final size 50,684, 38,695, 21,808, 12,680 |
| H5 | pass | every main-arm interval for the window ratio includes 1. The documentation arm at rho 0.5 and shape 0.05 does not: ratio 0.88, interval 0.76 to 0.99 |
| H6 | pass | P100ln / P100 = 0.49, below 0.7 |

## What this shows, and what it does not

A stable infectiousness trait that is not mirrored in exposure is interchangeable with redrawing that multiplier every episode. The simulator already had that result in the toy, and it survives schedules, natural history and the base heterogeneity of k about 5.

Coupling the same trait to exposure is a different epidemic at the same early R: fewer people are infected, because the people who are easy to infect are also the people who have already been infected. That is the signature the instructions describe as a stylised model of contact propensity with reciprocal exposure. It is not evidence that the trait is behavioural, and it is not a biological parameter. Only a gamma and a lognormal, and a single exponent, were tested.

The composition rule with the 20 to 50% uplift was fitted on uncoupled multipliers. It does not travel to the coupled arms. Using it to predict early k at rho > 0 would overstate k.

Cohort k is not a usable signature of persistence. The toy said so, and PanSim agrees. The mean trait of who gets infected is the signature that moves, and only at full coupling is the decline across a ten-week run larger than the replicate noise.

The two shape-0.05 arms exist to record a negative. At rho 0.5 or 1, early k stays well above 0.33 even at a very wide trait. Matching the published k of 0.3 with a persistent gamma trait requires weak coupling, around the P25k corner (shape 0.08, rho 0.25, early k 0.27), and that corner has a first-cohort mean trait of about 3.4 rather than 1.

## PS2, profile and calibration

One uncontrolled run with no multiplier, eight weeks, seed 7. Infectiousness is the value `isInfectious()` at the first timestep of each day since infection, averaged over agents still infected, on days with at least 30 such agents.

The profile rises from 0.48 on day 1 to a maximum of 0.87 on days 6 and 7, and is still 0.49 on day 19. The days at or above half the maximum are 1 to 19, and they hold 95% of the profile's area. The days at or above three quarters of the maximum are 3 to 16, and they hold 77%. Median time from infection to diagnosis is 6 days (10,282 diagnosed agents); median time to symptom onset is 5 days. The share of profile area before day 6, `f_pre`, is 0.23.

`f_pre` is below 0.5, so predictions Q2 and Q3 are expected to fail: the peak is not mostly before diagnosis. Because the half-maximum window already holds 95% of the area, a multiplier that acts only inside it is close to a multiplier that acts all the time. The shape solve agreed. Arm C, constant shape 0.3, has early k 0.301 at early R 1.33. The half-maximum arm reached early k 0.291 at shape 0.270. The three-quarter arm reached early k 0.257 at shape 0.177, inside the 15% band around 0.301.

| arm | coefficient | shape | window | calibration early R | calibration early k |
|---|---:|---:|---|---:|---:|
| A0 | 3.80e-4 | none | | 1.350 | |
| C | 4.80e-4 | 0.30, constant | all days | 1.326 | 0.301 |
| Pk50 | 4.75e-4 | 0.270, peak | days 1 to 19 | 1.295 | 0.291 |
| Pk75 | 4.45e-4 | 0.177, peak | days 3 to 16 | 1.357 | 0.257 |

## PS2 results

Eight replicates, seeds 2101 to 2108, twelve weeks, levers from day 21. Early R on the uncontrolled runs is 1.30, 1.32, 1.29 and 1.27 for A0, C, Pk50 and Pk75. The highest is 4% above the lowest, inside the 10% matching tolerance, and all are near the target of 1.3.

**The ranking of the levers does not change.** Isolating diagnosed people cuts cumulative infections by 16% under the constant multiplier and 17% under the half-maximum window. Isolating the household as well cuts them by 52% and 54%. The paired difference (peak minus constant) is +1.3 percentage points for diagnosed-only isolation (interval −1.2 to +3.7) and +2.0 points for household isolation (interval −0.7 to +4.8). Both intervals include zero. Household isolation remains far ahead of diagnosed-only isolation in every arm, including the narrower three-quarter window (55% against 18%). For lever comparisons in this natural history, the constant multiplier is enough.

That is what Q2 and Q3 said would happen when `f_pre` is small. It is 0.23. The top 10% of infectors transmit 89.0% of their infections before the infector is diagnosed under the constant multiplier and 89.0% under the half-maximum window. The rest of infectors are at 88.8% and 88.2%. There is no extra pre-diagnosis concentration to exploit.

| arm | early R | early k | generation interval mean (sd) | before diagnosis | top 10% before diagnosis | isolate reduction | household reduction | infections averted per 1,000 household quarantine-days | peak hospital, no lever |
|---|---:|---:|---|---:|---:|---|---|---:|---:|
| A0 | 1.295 | 4.57 | 5.50 (3.52) | 0.881 | 0.871 | 0.153 (0.144 to 0.163) | 0.510 (0.495 to 0.525) | 101 | 624 |
| C | 1.320 | 0.297 | 5.35 (3.56) | 0.889 | 0.890 | 0.160 (0.143 to 0.177) | 0.516 (0.488 to 0.543) | 101 | 694 |
| Pk50 | 1.291 | 0.263 | 5.36 (3.59) | 0.887 | 0.890 | 0.173 (0.145 to 0.200) | 0.536 (0.504 to 0.568) | 109 | 627 |
| Pk75 | 1.267 | 0.228 | 5.25 (3.65) | 0.889 | 0.881 | 0.182 (0.160 to 0.205) | 0.554 (0.513 to 0.596) | 120 | 538 |

Generation-interval quantiles are the same shape in every arm: 10th percentile 1 day, median 4 or 5, 90th percentile 10. About 57 to 58% of transmissions come from infectors who never leave the "well" states, 27% from before symptom onset, 9% within two days after onset, and 5% later. Those shares do not move between C and Pk50.

The correlation between an infector's offspring count and the mean generation interval of those offspring is 0.11 (0.10 to 0.12) under the constant multiplier and 0.10 (0.09 to 0.11) under the half-maximum window. Both exclude zero, so Q4's claim that the association appears only under the peak window fails. Under the narrower window the correlation is 0.38 (0.37 to 0.38). That arm was the sensitivity analysis, and it is the one place the timing of offspring and the number of offspring move together.

| id | verdict | numbers |
|---|---|---|
| Q1 | partial fail | multiplier mean 1.006 and CV² within 1% of 1/shape. Profile area 4.2% low, outside the 2% bar. Days 20 and 21 differ by 17% and 33% |
| Q2 | fail, as expected | top-10% pre-diagnosis share 0.890 against 0.890. `f_pre` is 0.23, not above 0.5 |
| Q3 | fail, as expected | peak-window reductions are slightly larger, not smaller, and the paired intervals include zero |
| Q4 | fail | the offspring–interval correlation excludes zero under both C and Pk50. It is much larger (0.38) under the narrower window |

Pk50's eight-replicate early k is 0.263 against C's 0.297, 11% lower. The two-seed calibration had matched them (0.291 against 0.301). The lever comparison is matched on early R. It is not matched quite as tightly on early k as the calibration was.

## Evidence tables

`study/ps/inputs/published_evidence_verified.csv` is the starter table with full-text checks. Thirteen of eighteen rows are verified. Not verified, and not cited as checked: Nikolay 2019 (E08, E09; the full text did not download), Pan 2026 (E11, E12; not open), Garg 2025 (E15).

Two starter rows change once the full text is read. Adam 2020 does report k: 0.43 (0.29 to 0.67) from the observed offspring distribution, and 0.33 (0.14 to 0.98) from the cluster-size sensitivity. The 19% (15 to 24%) share is confirmed. Marks 2021 states the odds ratio per log10 increase in viral load, in copies per mL. Table 3 gives an adjusted odds ratio of 1.29 (1.10 to 1.50); the abstract rounds it to 1.3 (1.1 to 1.5). The 12% attack rate is index viral load below 1e6 copies/mL, and the 24% is 1e10 or higher.

`published_k_from_R7.csv` is one row per study in Wegehaupt 2023, Tables 1 and 2. Where a study reports several populations, `which_estimate` names the row and `other_estimates` holds the rest. `published_k_by_setting_R6.csv` has both Chen 2025 tables. The cluster-ascertainment table is the one behind the abstract (retail 0.05, household 1.11). The individual-offspring table is a sensitivity analysis and is a different quantity; its household k is 126 with an interval from 18 to 280, which is not the cited household result.

The transcribed phase 2 grid in the instructions matches `summaries/p2_grid.json` at the reported rounding (baseline k 5.25 against 5.248, shape 0.3 k 0.33 against 0.332, shape 0.1 k 0.127 against 0.127).

Figures, each with a CSV and a script that refuses to write the figure if its title is no longer true: `study/figures/F2_published_k`, `F3_early_k`, `F5_ps1`, `F6_ps2`, `F7_lever_household`. F7 is supplementary and says so in the title: n = 5, shares from different experiments, one rank inversion (75.3% at household share 0.15 against 75.4% at 0.21). F4, the concentration curves, and F1, the schematic, were not drawn.

## Corrections to earlier notes

Section 5.1 of `13_phase3_results.md` read 184,745 agents against 9.2 million of residence capacity as a 1-in-50 sample of Hungary. That is withdrawn. The model is Szeged. The capacity field sums to 9.2 million and the median residence capacity is 92, against about two assigned agents, but that ratio is not a sampling fraction. The paragraph in 13 has been corrected, and the banner at the top of 13 points here.

## Caveats

Gamma and lognormal only, and one coupling exponent. The trait is not a biological parameter. Reinfection happens and the trait persists across episodes; the trace keeps the latest episode, so a reinfected agent's infection day in the trace is the later one. Cohort windows after day 30 therefore miss some people whose first infection was earlier. One city. The toy numbers in the instructions are not targets. P100ln is not matched on early R in the twelve-replicate table. Pk50's eight-replicate early k is 11% below the constant-multiplier arm even though the two-seed calibration had matched them. The trace keeps one row per agent, the latest episode.

## How to rerun

From `superspreader/study/ps`, with the GPU build at `build_gpu/panSim`:

```
python3 test_t1.py
python3 test_t2_t4.py
python3 regression.py
python3 run_ps1.py all
python3 analyze_ps1.py
python3 run_ps2.py all
python3 analyze_ps2.py
python3 make_figures.py
```

Finished jobs are skipped. `run_ps1.py core` stops after the four-replicate pilot if H1 fails. Calibration caches are `study/results/ps/cal/*/k.json`.
