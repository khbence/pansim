# ACT-049 review of 06_study_results.md, and next steps

Reviewer pass over the simulation study, 10 October 2026. Written against 06_study_results.md, 04_pansim_experiments.md and 01_literature_review.md. No PanSim runs were made for this review; the checks below are arithmetic and sampling calculations on the numbers reported in 06, plus algebra on the transmission kernel as that file documents it. Where I say "I could not check", the raw traces and summaries under `superspreader/study/` are not in the ACT-049 folder.

## 1. Verdict

The study is well instrumented, honestly reported and, as far as I can test it, arithmetically correct. Every figure I recomputed from the tables matched: the population scaling (one agent for 55.8 people, hospital peaks 144 and 225, prevalence peaks 2,980 and 3,960), every percentage reduction and averted-per-person-day figure in experiment 2, the pooled first-week reproduction numbers, the pure-backward inflation factors, and the Poisson zero-probability check in experiment 4. The p80 column is consistent with the reported (R, k) pairs under a negative binomial to within 0.01 in all seven cells I tested. The instrumentation design is good: the off-by-default flags, the empty closure file, the per-replicate venue concentration, and the decision to state the moment estimator and its infinite-k convention.

Three conclusions in the note need to be changed, one of them substantively.

1. **The attribution of the baseline tail to occupancy is wrong, and the kernel makes it impossible.** Section "Experiment 1" says the k of 4.5 is "the occupancy distribution, not who shares a household". Under the documented hazard that cannot be true, and the shuffle does not test it.
2. **Experiment 2 did not test its hypothesis**, so "the proposed ranking is not supported" should become "not tested". The note half says this and then draws the conclusion anyway.
3. **Experiment 4's headline comparison is confounded by mean offspring**, and the finding that heterogeneity does not raise the zero share reverses once the arms are matched. The absence of a 50-case chain is not a thin-sample statement.

Everything else in the note stands, including the two results I regard as the most valuable: the multiplier behaves exactly as the options table predicted, and the detection audit closes a factor of about two, not the factor of 26 that would be needed.

## 2. The kernel result that reframes most of the study

The note gives the per-step hazard as `1 - exp(-c * density * I_loc * eventBoost * seasonality * dt)` with `density = (sum of agent infectiousness present) / (number present)`.

Take one infector with infectiousness `I_i` in a venue holding `N` people, of whom a fraction `s` are susceptible. Its expected offspring at that venue in one step is

    N*s * (1 - exp(-c * (I_i/N) * I_loc * dt))  ~  s * c * I_i * I_loc * dt

The `N` cancels. **Expected offspring does not depend on occupancy at all.** Two further consequences follow:

- The offspring count at a venue is Binomial(N-1, p) with `N*p` held fixed, so its variance is `N*p*(1-p)`, which is at most Poisson and is strictly below Poisson in small venues. Occupancy heterogeneity therefore cannot generate overdispersion in this kernel. It can only reduce it.
- Any overdispersion the model does produce must come from between-agent variation in the integral of own infectiousness times venue infectiousness times time present: that is, **natural history (infectious period, asymptomatic fraction, infectiousness trajectory) and the time each agent allocates across location types.** Not occupancy, not household membership.

The shuffle is consistent with this and does not discriminate: permuting place assignments inside each location type preserves both the occupancy multiset and each agent's time allocation across types, so it changes nothing that the kernel can see. The correct null result from that cell is "co-membership does not matter", which is what the note's own numbers show (k 4.56 to 4.39). The sentence attributing the tail to occupancy should be deleted.

This single fact explains the study's main negative results together:

- Venue amplification raises the venue concentration v10 from 0.48 to 0.84 while k only falls from 4.6 to 0.54, because a venue multiplier raises everybody's hazard in that venue equally. It concentrates *places* efficiently and *people* badly.
- Explicit events are "almost invisible" at probability 0.02 and factor 15. The note blames occupancy dilution, which is right, and the algebra above says the dilution is exact, not partial.
- A Skagit-type event, one infector and 53 secondary cases, is not representable at any plausible venue multiplier, because expected offspring is capped by `c * I_i * I_loc * dt` regardless of how many people are in the room.
- Occupancy caps, the intervention that [R30] found outperforms uniform mobility reduction, are close to a no-op in this kernel by construction. The note observes that `communityScale` is not a cap. The deeper point is that a cap could not do much even if it were implemented.

**I have derived this from the prose description of the kernel in 06, not from the code. Please confirm against the implementation before acting on it.** A one-run diagnostic settles it: hold everything fixed and double the occupancy of one location type (or halve the number of venues of that type). If mean offspring per infector is unchanged, the algebra holds.

## 3. Experiment-by-experiment

### Experiment 1 (variance partition)

Sound apart from the attribution above. Two further points.

**Multiplier composition is close but not exact.** If independent gamma factors compose, squared coefficients of variation add: `CV2_total = CV2_shape + CV2_base + CV2_shape*CV2_base`. With a baseline k of 4.56 that predicts a total k of 0.695, 0.233 and 0.081 for shapes 1, 0.3 and 0.1. The run gives 0.90, 0.29 and 0.11, consistently 25 to 35 percent higher. So the multiplier does not simply pass its shape through, and the ablation arithmetic should not be treated as exact. Depletion or the 14-day censoring is the likely cause.

**The depletion confound is present but probably small.** Cells differ in final size by a factor of four (37,000 against 145,000), and depletion truncates the offspring tail, which raises k. That makes the cross-cell k comparison formally unsafe. In the study's favour, the shape-0.1 pair moves from 5,358 infections (k = 0.11) to 83,621 infections (k = 0.094) with k if anything falling, which suggests the bias is modest here. I would still want one matched-R control before the venue conclusion is quoted outside the project.

**The missing control.** There is no cell in which schedule heterogeneity is actually switched off. Give every agent the population-mean time allocation across location types, keep natural history as it is, and k should rise toward its natural-history-only floor. Pair that with a fixed-infectious-period cell to find the floor itself. Those two cells, not the shuffle, are what decompose the baseline 4.56.

### The fitted first two waves

No objection. The baseline sits on the scaled hospital peaks, the three heterogeneity corners miss them by factors of 7 to 10 in the first wave, and the note is explicit that the coefficient was never re-fitted. That is the correct finding and the correct caveat. It also means **no heterogeneity setting in this study is currently usable in a production run**, which should be stated that plainly in the brief.

### Experiment 3 (detection audit)

The most solid of the four. The direction matches the tracing literature [R9], the magnitudes are reported honestly, and the conclusion that detection moves k by a factor of about two rather than 26 is well supported by the tables.

Two caveats to add. First, the observation models are plausible inventions, not reproductions of any published estimator; the mixed-backward row returning infinite k suggests it does not resemble how, for example, Adam et al. built clusters. Replicating one published estimator end to end on simulated trees would be a stronger audit. Second, and not mentioned in the note: published k near 0.1 was mostly estimated in exponentially growing, undepleted epidemics with R near 2.5, while these runs estimate k at a censored R near 1.0 to 1.4 in a depleting population. That is not a like-for-like comparison. The cheap fix is one run at a coefficient giving R near 2.5, with k measured on the first one or two generations only.

The home-versus-community split (home k above community k in every trace) is a genuinely useful result and matches the direction of the Hong Kong contrast in [R6]. Keep it.

### Experiment 2 (same k, different mechanisms)

**This experiment did not test its hypothesis and its conclusion should be withdrawn.** The protocol in 04 asked for arms matched on early generations. The executed match was on eight-week censored mean offspring, which in set II is held near 1 only because the arm has already burned through the population. The arms differ by 3.5-fold in first-week reproduction (0.95 against 3.36) and by 70-fold in day-14 prevalence. Set I is subcritical and never takes off; set II is supercritical and saturates. Any intervention comparison between them is a comparison between "epidemic that was never going to happen" and "epidemic that already happened", not between biology and setting.

The note sees this, then says "the comparison is still the one that decides whether the split matters" and concludes that the proposed ranking is not supported. It does not follow. What the runs show is that a saturating epidemic responds to almost nothing, which is a statement about saturation.

The levers are also not matched on intensity or cost. Halving community infectiousness from day 0 is a far larger intervention than quarantining diagnosed individuals, and the two arms spend quarantine person-days differing by a factor of 50. Ranking levers requires matched cost.

### Experiment 4 (subcritical chains)

The arms are matched on generation-1 R and not on the quantity that drives chain length. Implied all-node mean offspring is `1 - 1/(mean chain size)`:

| arm | mean chain | implied all-node mean offspring |
|---|---:|---:|
| flat, target 0.30 | 1.50 | 0.333 |
| flat, target 0.20 | 1.33 | 0.248 |
| caregiving, target 0.30 | 1.98 | 0.495 |
| caregiving, target 0.20 | 1.52 | 0.342 |

The "clean pair" has generation-1 R of 0.231 and 0.250, but all-node mean offspring of 0.248 and 0.342, a 38 percent gap. Chain length is driven by the all-node mean, so the contrast confounds mean with heterogeneity.

I resimulated the two arms as branching processes at their own reported parameters, 52 chains per arm, 4,000 repetitions, which gives the sampling spread the note does not report:

| arm as reported | max chain, median (90% interval) | P(max >= 13) | zero share, median (90% interval) |
|---|---|---:|---|
| flat, Poisson mean 0.246 | 4 (3 to 7) | 0.000 | 0.783 (0.705 to 0.869) |
| caregiving, NB mean 0.342, k 0.70 | 7 (4 to 14) | 0.083 | 0.762 (0.688 to 0.844) |
| control: Poisson at the caregiving mean 0.342 | 6 (3 to 10) | 0.013 | 0.714 (0.634 to 0.803) |

Three things follow.

1. **The observed max chain of 13 is a 92nd-percentile draw even for the caregiving arm**, and a Poisson process at the same mean already gives a median of 6. Most of the 5-to-13 jump is the higher mean, not the heterogeneity. A single max-chain value from 52 chains is weak evidence either way.
2. **The zero-share conclusion reverses once the arms are matched.** The note concludes that heterogeneity does not raise the share of cases with no offspring, from 0.768 against 0.747. At the caregiving arm's own mean, Poisson gives 0.714 and the reported heterogeneity gives 0.762. Heterogeneity does raise the zero share, by about 5 points; the observed non-effect is the higher mean cancelling it. The note's explanation ("the extra variance was paired with a slightly higher mean") is right, but the stated conclusion contradicts it and should be rewritten.
3. **The observed differences are inside sampling noise.** The zero shares differ by 0.021 against 90% intervals roughly 0.08 wide either side. As reported, experiment 4 does not resolve its hypothesis.

**On the missing 50-case chain.** This is not a thin-sample statement. At R = 0.33 I get a single-seed probability of a chain reaching 50 of essentially zero at k = 0.7, 0.3 and 2, and 0.01 percent at k = 0.1. More replicates would not have produced one. The reason is that the achievable k is far from the Nipah target, which brings me to the most useful number in this review.

**What the published Nipah signature actually demands.** A 92 percent zero-offspring share at R = 0.33 requires a negative-binomial k of about **0.036**, at which point the average transmitter infects about 4.1 people. At that k the probability of a chain exceeding 50 is 0.068 percent per seed, or 0.34 percent from 5 seeds, the same order as the 1.1 percent in [R22]. The caregiving profile reached k = 0.64 to 0.70, roughly **20 times too large**, and the hardest venue setting anywhere in the study (strength 8) reached 0.54. So the Nipah-like scenario as configured cannot reproduce the empirical signature, and the gap is in the dispersion this kernel can generate, not in the number of replicates. That is a much stronger and more useful statement than "no chain reached 50", and it is the same wall the SARS-CoV-2 experiments hit at k near 0.1.

## 4. What I now conclude across the study

1. **The kernel, not the parameter settings, is the binding constraint.** A frequency-dependent hazard with occupancy in the denominator cannot convert crowding, venue size or schedule structure into individual offspring overdispersion. Venue multipliers concentrate places but not people. Everything the study found about settings follows from this, and none of it is evidence about the world.
2. **The literature review's best-supported explanation is the one PanSim currently cannot express.** Section 7 of 01_literature_review.md grades setting and event effects as "supported" for SARS-CoV-2 on the strength of ventilation, duration and occupancy evidence [R6, R10, R11, R30]. Those are exactly the mechanisms this kernel neutralises. The honest position for the brief is that PanSim can currently represent the biological explanation and only a weak form of the setting explanation.
3. **The per-agent multiplier works as advertised and is the only route to a published k today**, but it is a reduced-form device: it buys the k and buys nothing mechanistic, it shrinks the epidemic because the median gamma draw is below the mean, and it cannot be dropped on the fitted coefficient.
4. **Detection bias is a real factor of about two and does not close the gap.** That conclusion survives review.
5. **The identifiability warning in 03 is strengthened, not weakened.** Matching one k, especially a k averaged over a depleted wave, pins neither the split nor early growth, which experiment 2 demonstrates accidentally.

## 5. Next steps, in priority order

**N1. Settle the kernel question (blocking, cheap).** Confirm from the code whether expected offspring is independent of occupancy, and run the doubling diagnostic in section 2. If confirmed, cost a partially density-dependent option: `density = sum(I) / N^alpha` with alpha between 0 and 1 as a run flag, or a volume and ventilation denominator in the style of the quanta model in [R11]. Then re-run one experiment-1 column at alpha = 0.5 and 0 to see whether occupancy heterogeneity starts producing offspring overdispersion. This decides whether PanSim can represent superspreading events at all, and it affects every existing fit, so it is a decision for the PanSim owners and not something to slip in.

**N2. Re-do the variance partition with the controls that isolate the factors.** Add (a) a true schedules-off cell with population-mean time allocation across location types, (b) a fixed natural-history cell to find the k floor, (c) per-cell recalibration of the infection coefficient to a matched early R, and (d) k measured on generation 1 or the first two weeks rather than a censored window. Report the decomposition as squared coefficients of variation, which add, rather than as k, which does not.

**N3. Re-run experiment 2 to its own protocol.** Calibrate both arms by bisection to the same early growth rate (suggest generation-1 R near 1.3, both arms supercritical but neither saturating inside the window), trigger each intervention at a fixed prevalence rather than on day 0, and match levers on cost (equal quarantine person-days, or equal fractional hazard reduction). Only then is the rank-reversal question answerable. Until it is run, the brief should say the ranking is untested.

**N4. Re-aim experiment 4 at k, not only at R.** Calibrate jointly on generation-1 R and on the zero share or k, with k = 0.036 at R = 0.33 as the Nipah-like target from [R18, R21, R22]. Match arms on all-node mean offspring. Raise to about 1,000 chains per arm if the far tail is to be claimed at all, and report intervals on every k and zero share. If the kernel cannot reach k = 0.036 without an imposed multiplier, that is the finding, and it should be stated as a model limitation rather than as a property of Nipah.

**N5. Do the calibration trade the study deliberately skipped.** Re-fit the infection coefficient jointly with the chosen heterogeneity parameters against the two Hungarian waves, so that at least one heterogeneity configuration preserves the hospital fit. Until this exists there is no usable overdispersion setting in PanSim, and experiments 1 to 4 remain diagnostics rather than model options.

**N6. Reporting hygiene.** Add uncertainty intervals to every k, zero share and venue concentration figure; the experiment-4 contrasts as reported are inside noise. Keep replicate counts and node counts in the tables, since k from about 80 nodes is nearly uninformative.

**N7. Document updates once N1 is settled.** 03_pansim_options_table.md needs a revision note: option A is confirmed, option B is refuted on the current kernel and conditional on N1 thereafter, option C's calibration order needs N5 inserted first, and option D should be marked as requiring density dependence to be worth building. I can prepare that revision on request.

## 6. What I would like, if it is easy to share

1. The transmission-kernel source, or confirmation of the `density = sum(I)/N` form, to turn section 2 from algebra into a checked statement.
2. `superspreader/study/summaries/` and the per-node offspring counts for the experiment-4 arms, so uncertainty intervals can be computed from the real trees instead of from a branching-process reconstruction.
3. Replicate counts and node counts behind the experiment-4 zero shares.
4. Whether `locations0.json` carries any venue capacity or area field. If it does, a density-dependent denominator is much cheaper to add than N1 assumes.

## 7. Method note for this review

Checks run: recomputation of every percentage, ratio and scaling figure in 06; a negative-binomial simulation of the p80 statistic against seven reported (R, k) pairs; the gamma composition identity for the multiplier cells; a branching-process resimulation of the two experiment-4 arms at their reported parameters (52 chains, 4,000 repetitions) for sampling spread; chain-size tail probabilities at R = 0.33 across k; and a root solve for the k implied by a 92 percent zero share at R = 0.33. The kernel algebra in section 2 is derived from the description in 06 and has not been checked against code. No PanSim runs were made.
