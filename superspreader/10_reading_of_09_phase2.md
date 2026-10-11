# ACT-049 reading of 09_phase2_results.md

10 October 2026. Checks and interpretation of the phase 2 runs. No PanSim runs were made for this note; everything below is recomputation against the tables in 09 plus branching-process calculations. Three readings differ from the note's own.

## 1. Verification

Internally consistent everywhere I tested. For all ten rows of the density-exponent and heterogeneity grids, the reported zero share and top-20% share match what a negative binomial with that row's (R, k) gives, to within 0.02. The per-1000-person-day figures in section 4 reproduce exactly (108, 43, 77, 101). The Poisson zero check in section 6 reproduces. The calibration-to-early-R protocol fixes the design flaw that made 06 experiments 1 and 2 uninterpretable, and the early-window k values are the right statistic.

**One reporting fix.** The CV2 column in section 3 is the squared coefficient of variation of *realised* offspring, equal to `1/R + 1/k`, which I confirmed on four rows (0.90, 1.86, 3.76, 8.63 against 0.90, 1.87, 3.78, 8.61). That includes Poisson sampling noise. At the baseline row, 0.71 of the 0.90 is Poisson and only 0.19 is heterogeneity. The column therefore cannot be used for the variance partition it was added for; the additive quantity across mechanisms is `1/k` alone. Easy to fix, worth fixing before anyone reads the column as a decomposition.

## 2. Track A is settled, and the answer is negative

Removing the occupancy denominator entirely (alpha = 0) moves early k only from 5.25 to 1.79. At alpha = 0.5 it reaches 2.66, and the median occupancy at the moment of infection is already 649 people. A regime in which half of all infections happen in venues holding 649 or more is not a description of Hungary. So density dependence buys roughly a factor of two in k at the cost of an implausible venue distribution, and it does not approach the published range.

That closes the kernel question raised in 07 and 08. It is a clean negative result and it should be stated as one: **no plausible parameterisation of this kernel, with or without the density exponent, generates published-range dispersion from venues, occupancy or schedules. The per-agent multiplier is the only mechanism in PanSim that reaches it.**

The uncomfortable corollary deserves to be in the brief. Section 7 of 01_literature_review.md grades setting and event effects as the best-supported explanation for SARS-CoV-2 and the biological explanation as mixed. PanSim can express the weaker-supported one and not the better-supported one. That is a limitation of the model, not a finding about the world, and section 4 of this note shows the choice between them changes intervention rankings.

## 3. The intervention hypothesis is now supported, which 09 does not say

This is the third pass over the same question and the conclusion has moved twice:

- 06 experiment 2: "the proposed ranking is not supported".
- 07: not supported is wrong, the arms were never matched, so it is **not tested**.
- 09 section 4, with arms matched to within 10% on early R and paired seeds: the ranking predicted in 04_pansim_experiments.md **is reproduced**.

The prediction in 04 was that a biology-dominated arm favours individual-targeted measures and responds weakly to venue controls, and a setting-dominated arm does the opposite. The runs give exactly that reversal:

| | Set I, biology (shape 1) | Set II, setting (venue strength 8) |
|---|---|---|
| close venue types | 37.2% | 75.3% |
| isolate diagnosed + household | 53.3% | 34.2% |

Household isolation beats venue closure by 16 points under person-level heterogeneity and loses to it by 41 points under setting-level heterogeneity. 09 reports this as "the ranking depends on where the heterogeneity sits" and does not connect it to the hypothesis it was built to test. It should, because it is the one experiment in the whole packet that confirms a prior prediction rather than refuting one.

Two qualifications that belong next to it. Masks, a uniform non-home hazard cut, are top or joint-top in both arms (74% and 79%), so the reversal is specifically venue closure against household isolation, not "targeted versus untargeted". And isolation of diagnosed agents alone is the weakest lever in both arms (17%, 23%), so the "individual-targeted" half of the prediction holds only when the household goes with the case. The arms also differ 2.6-fold in uncontrolled final size (54,784 against 20,928) at early R of 1.30 and 1.17, so only the within-arm orderings are safe; the cross-arm percentage levels are not.

## 4. The result 09 under-reads: k does not predict tail risk

Section 6 reports that chains over 50 cases appear only in the caregiving plus shape-0.09 arm, and explains it as "without a multiplier the kernel's offspring variance is at most Poisson, so only the per-agent multiplier does". That explanation does not fit the arms being compared, because two of them have a multiplier:

| arm | all-node k | chains > 10 (of 3808) | chains > 50 | max chain |
|---|---:|---:|---:|---:|
| flat, shape 0.04 | **0.054** | 51 | **0** | 45 |
| caregiving + shape 0.09 | **0.088** | 22 | **3** | 116 |

The arm with the *lower* k, that is the more overdispersed one, produced no long chains. Under a negative-binomial branching process with each arm's own reported mean and k, the ordering should be the reverse: the flat shape-0.04 arm should yield about 1.2 to 1.5 chains over 50 and the caregiving arm about 0.3 to 0.4.

Testing the two arms against their own marginals:

- Flat shape 0.04 behaves like its k in the mid-tail: 51 chains over 10 observed against 46 expected, one-sided p = 0.23.
- Caregiving plus shape 0.09 does not: 22 chains over 10 observed against 41 expected, one-sided p = 0.0008.

So the caregiving arm's chain-size distribution is a genuinely different shape, **thinner in the middle and heavier at the extreme than any negative binomial with the same k**. That part is statistically solid. The headline "0 against 3 chains over 50" on its own is not: Fisher exact p = 0.25, and observing zero in the flat arm when 1.35 are expected has p = 0.26 by itself.

Two mechanisms would produce this, and the runs cannot yet separate them.

1. **Generation-correlated infectiousness.** The caregiving profile sets home 2.5, hospital 4, nursing 3, church 3 and 0.35 for everything else. A chain that reaches a high-infectiousness setting stays there, so a case's offspring inherit their parent's setting. Branching processes with independent offspring cannot generate that, and the marginal k does not see it.
2. **Shorter realised generation interval.** Caregiving settings are daily and immediate, so more generations fit inside the 7-week horizon, and the flat arm's long chains may simply be truncated.

Both are checkable from traces already on disk, at no GPU cost: per arm, the mean realised generation interval, the number of generations per chain, and the correlation between a case's location type and its parent's. If it is mechanism 1, the finding is substantive.

Why it matters either way: **calibrating PanSim to a published k does not calibrate its large-outbreak risk.** This sharpens the identifiability warning in 03_pansim_options_table.md from "the split is not identified" to "the summary statistic everyone calibrates to is not sufficient for the outcome we care about". It also partly restores the setting explanation, through a channel the dispersion parameter cannot measure, which is worth saying given section 2 above. And it bears directly on the Nipah scenario: the 1.1% chance of an outbreak over 50 cases from 5 seeds in [R22] is a tail statistic, so matching k alone will not reproduce it, whereas a nosocomial and caregiving structure might.

## 5. The practical answer to N5, which is buried in section 7

Section 7 is the one that decides whether any of this is usable, and it has a clear winner that the note does not name. Refitting the infection coefficient to the wave-1 hospital area succeeds for every corner, so **wave 1 does not identify k at all**. The wave-2 hold-out does discriminate. Welch tests on the reported means and SDs, 4 replicates each, against the baseline's 222:

| corner | wave-2 peak | difference | p | verdict |
|---|---:|---:|---:|---|
| multiplier shape 0.3 | 252 (57) | +30 | 0.38 | consistent |
| multiplier shape 0.1 | 152 (36) | -70 | 0.022 | differs |
| covid venue strength 1 | 148 (18) | -74 | 0.001 | differs |
| venue strength 4 | 120 (29) | -102 | 0.002 | differs |
| shape 0.3 plus venue 1 | 105 (33) | -117 | 0.002 | differs |

**Multiplier shape 0.3 is the only configuration that survives the hold-out**, and it is also within 1% of the baseline on total infections where the others lose 25% to 37%. Its early k is 0.32 (0.28 to 0.37). So the answer to the question that has been open since 06 is: PanSim can carry dispersion of about 0.3 with the hospital fit intact, and cannot carry 0.1.

That value is defensible against the literature. It sits inside the range in [R7] (0.04 to 2.97, with 93% of studies below 1) and inside the setting-stratified range in [R6] (0.05 to 1.1). It is not the 0.1 of the headline early estimates in [R3].

Caveats to carry with it: 4 replicates, a wave-1 area target itself known only to about 15%, and shape 0.3's wide wave-2 spread (SD 57, range 198 to 329) means "consistent" partly reflects low power rather than a good match. This is the only corner not rejected, not a corner shown to be right.

## 6. The detection audit changed sign, and that makes the gap worse

06 concluded that detection moves k by about a factor of two. At matched early R of 2.5, section 5 of 09 finds up to a factor of 17. The important part is the direction: **every sampling scheme inflates k**, that is, makes the epidemic look less overdispersed than it is. Backward tracing raises a true 0.13 to between 1.4 and 2.3; symptom-forward sampling reached 0.97 in one replicate against a truth of 0.14; and in the flat arm every non-true scheme returns infinity.

Three consequences, of which 09 draws only the first.

1. Sampling cannot turn a true k of 5 into an observed 0.1. The kernel gap stands, and it is not an artefact of how cases are found.
2. Published k values derived from backward-traced clusters are **upper bounds**. If anything the true dispersion is below the published 0.1, so the distance between PanSim and reality is larger than the headline numbers suggest, not smaller.
3. Cluster sampling was close to unbiased in the shape-0.1 arm (0.086 to 0.166 against a truth of 0.120 to 0.137). So PanSim should be compared against cluster-derived published estimates, such as [R4] and [R6], and not against tracing-derived ones.

## 7. What remains open, in priority order

1. **Separate the two explanations for the caregiving tail** (section 4). Generation intervals, generations per chain, and parent-child location-type correlation, all from existing traces. No GPU. Do this first, it is the cheapest open question with the largest interpretive payoff.
2. **More chains for the tail claim.** 3 events out of 3808 will not support a statement about outbreaks over 50 cases. If the tail is to be used for the Nipah scenario, it needs roughly an order of magnitude more chains in the two shape arms.
3. **Confirm shape 0.3 as the production setting** with more replicates on the wave-2 hold-out, since the current verdict rests on 4 replicates and a wide spread.
4. **Set II lever estimates at two other venue strengths**, as 09 itself recommends, before the rank reversal in section 3 is used in any decision.
5. **The base k of 5.25 is still undecomposed** between time allocation and natural history, since `--meanSchedules` and `--fixedInfectiousness` were not built. The reasoning for dropping them is fair and the practical conclusion does not depend on it, so this is last.

## 8. Documents to revise

- **03_pansim_options_table.md**: option A is now the only viable route and should be marked so, with shape 0.3 as the recommended setting; option B is refuted on this kernel and on the density-exponent variants; option D needs the section 4 finding, since setting structure affects tail risk even where it barely moves k. The identifiability note should be upgraded to say that k is not a sufficient statistic for large-outbreak risk.
- **02_brief.md**: section 5 should name shape 0.3 and early k 0.32 as the configuration that preserves the hospital fit, and should state plainly that the explanation the review grades best-supported is the one the model cannot express.
- **09_phase2_results.md**: the CV2 column definition (section 1 here), the attribution of the long chains to the multiplier rather than to the caregiving structure (section 4 here), and the two extra consequences of the audit (section 6 here).

## 9. Method note

Checks run: negative-binomial recomputation of zero share and top-20% share for all ten grid rows; identity check on the CV2 column; recomputation of the per-1000-person-day figures; negative-binomial branching simulations (400,000 chains per parameter set) for chain-size tails at each arm's reported mean and k; Poisson and Fisher tests on the chain counts; Welch tests on the wave-2 hold-out means and SDs as reported. No PanSim runs, and no access to the raw traces or summaries.
