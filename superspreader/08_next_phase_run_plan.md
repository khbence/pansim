# ACT-049 next-phase run plan (both tracks in parallel)

Follows 07_review_of_06_and_next_steps.md. Decision taken 10 October 2026: run the kernel investigation and the experiment re-runs together. The runners take one GPU job at a time, so the queue below interleaves them, puts the zero-GPU work first, and places the cells whose design depends on the kernel answer last. Flag names in backticks are the ones already in the study harness; flags marked NEW do not exist yet.

## Track A: the kernel

**A0. Zero-GPU check on traces already on disk (do this first).** The occupancy question can be answered without a new run. Take an existing `--infectionTrace` from the flat uncontrolled baseline, join each infection event to the venue where it happened, and bin venues of a single type (short shop and long visit both have hundreds of sites) by occupancy. Then plot, for each bin, mean secondary infections per infector-visit. If that curve is flat in occupancy, expected offspring is occupancy-independent and section 2 of the review holds. If it rises, the algebra is wrong and track A stops here. This is an afternoon of analysis against data that already exists.

**A1. Read the kernel.** Confirm from source whether the denominator is the head count present. Note whether `locations0.json` carries capacity, floor area or volume. If it does, a density-dependent denominator is much cheaper than assumed.

**A2. Confirmatory run (1 GPU job).** Halve the number of venues of one location type so occupancy roughly doubles at matched total visits, hold everything else fixed, 8 weeks, 2 replicates. Compare mean offspring per infector at that type against the baseline. Expected under the current kernel: no change.

**A3. Density exponent as a flag (NEW `--densityExponent`).** Replace the denominator with `N^alpha`, alpha in [0,1], defaulting to 1 so every existing fit is untouched. Then run one column of experiment 1 at alpha = 1, 0.5 and 0, shedding off, venues off, real schedules, 8 weeks, 4 replicates. The question is whether occupancy heterogeneity starts producing offspring overdispersion, and at what cost in realism. Report k, v10 and final size.

**A4. If A3 is promising, cost the quanta variant.** A volume and ventilation denominator in the style of [R11] rather than a bare exponent. Scope only; do not build it in this phase.

## Track B: the experiment re-runs

Two of these are blocked on nothing and should start immediately. B4's design depends on A3 only for the extra alpha arm, so run its alpha = 1 version now.

**B1. Variance partition with real controls (N2).** Add:
- NEW `--meanSchedules`: give every agent the population-mean time allocation across location types. This is the true schedules-off control. The venue shuffle is not.
- NEW `--fixedInfectiousness`: constant infectious period and infectiousness trajectory, to locate the floor of k.
- Per-cell recalibration of the infection coefficient so that every cell shares one early reproduction number. Do not compare k across cells at different force of infection.
- Measure k on generation 1, or on index cases infected in the first two weeks, not on a censored window of a depleting epidemic.
- Report the decomposition as squared coefficients of variation, which add, rather than as k, which does not.

Cells: {real, meanSchedules} x {natural history on, fixed} x {venues off, covid strength 1}, 8 weeks, 4 replicates. Eight cells plus the calibration runs.

**B2. Detection audit at a growing epidemic (N3 addendum, cheap).** The published k near 0.1 was estimated in undepleted epidemics at R near 2.5; the audit compared against a censored R of 1.0 to 1.4. Bisect the coefficient to an early R near 2.5 with venues off, start from about `0.00041 * 1.8`, then re-run the five observation models on the first two generations only. This is four or five short calibration runs plus one trace run, and it either strengthens the existing conclusion or qualifies it.

**B3. Experiment 2 to its own protocol (N3).** Bisect both arms to the same generation-1 reproduction number, suggest 1.3 so that both are supercritical and neither saturates inside eight weeks. Verify the match on generation-1 R and on day-14 prevalence before running any intervention cell. Trigger each lever at a fixed prevalence rather than on day 0, and match levers on cost: equal quarantine person-days, or equal fractional hazard reduction. Only then is the rank-reversal question answerable. Until this lands, the brief should say the ranking is untested, not unsupported.

**B4. Experiment 4 aimed at dispersion, not only at R (N4).** Three changes:
- Calibrate jointly on generation-1 R and on dispersion. The Nipah-like target from [R18, R21, R22] is R = 0.33 with about 92 percent of cases infecting nobody, which corresponds to k near 0.036.
- Match arms on all-node mean offspring, which is `1 - 1/(mean chain size)`, not only on generation-1 R. The current "clean pair" differs by 38 percent on that quantity, which is what drove the chain-length contrast.
- Raise to about 1,000 chains per arm if anything is to be claimed about the far tail, and report intervals on every k and zero share.

Starting points for the multiplier, from the composition identity with the observed 25 to 35 percent gap folded in:

| target | base arm | required gamma shape | suggested bracket |
|---|---|---:|---|
| k = 0.036 (Nipah-like) | caregiving profile, base k about 0.7 | 0.092 | 0.065 to 0.092 |
| k = 0.036 (Nipah-like) | flat, base k about 6.9 | 0.041 | 0.029 to 0.041 |
| k = 0.10 (SARS-CoV-2) | flat, base k 4.56 | 0.125 | 0.087 to 0.125 |
| k = 0.10 (SARS-CoV-2) | covid profile, base k 2.68 | 0.143 | 0.100 to 0.143 |

Re-bisect the coefficient after setting the shape, because the multiplier depresses the epidemic. If the kernel cannot reach k = 0.036 without an imposed multiplier, that is the finding, and it belongs in the brief as a model limitation rather than as a statement about Nipah.

**B5. The calibration trade the study skipped (N5).** Re-fit the infection coefficient jointly with the chosen heterogeneity parameters against the two Hungarian waves, so that at least one heterogeneity configuration keeps the hospital peaks. Until this exists, none of the settings in 06 is usable in a production run and experiments 1 to 4 remain diagnostics. This is the longest job in the queue, 37 weeks per evaluation, so start the bisection early and let it run behind the short cells.

## Suggested GPU queue order

1. A0 and A1 (no GPU).
2. B1 and B3 calibration bisections, and B2 calibration. Short runs, 2 weeks, 2 replicates each.
3. A2 confirmatory run.
4. B1 main grid, B2 trace run.
5. B3 intervention cells once the match is verified.
6. B4 calibration then the 1,000-chain grid at alpha = 1.
7. B5 joint re-fit, running behind everything else.
8. A3 density-exponent column, and the A3-conditional repeats of B1 and B4 if alpha turns out to matter.

## Reporting changes that apply to all of the above

Every k, zero share and venue-concentration figure gets an interval. Keep replicate counts and node counts in the tables: k estimated from about 80 nodes, as in the present experiment 4, is close to uninformative. State the estimator and whether the window was depleted.

## Documents to revise when the runs land

- 03_pansim_options_table.md: option A confirmed; option B refuted on the current kernel and conditional on A3 thereafter; option C needs B5 inserted ahead of its calibration order; option D marked as requiring density dependence to be worth building.
- 02_brief.md: section 5 should say which heterogeneity configuration, if any, preserves the hospital fit.
- 06_study_results.md: the three corrections in section 1 of the review.
