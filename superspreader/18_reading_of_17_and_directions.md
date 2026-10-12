# ACT-049 reading of 17_ps1_ps2_results.md, and directions forward

12 October 2026. Checks and interpretation of the PS1 and PS2 runs and the new evidence tables. No PanSim runs were made for this note; everything is recomputation from the delivered CSVs and JSON, plus branching-process and estimator simulations. Section 2 reports an error of mine that the full-text checks exposed.

## 1. Verdict

This round is clean. The agent ran the full PS1 grid, reported T6 as a partial failure instead of smoothing it, recorded that the infection coefficient moved opposite to my predicted direction and explained why, and corrected the 1-in-50 misreading in 13. Every figure of mine that I could recompute matched.

Checks that passed:

- **Profile arithmetic.** From `17_ps2_profile.json` I recomputed peak 0.8650 on day 7, the 50% window as days 1 to 19 with f_peak 0.9494, the 75% window as days 3 to 16 with f_peak 0.7683, and f_pre 0.2269. All four match the reported values exactly.
- **Offspring distributions are negative-binomial in shape.** For seven of the eight PS1 arms, the reported zero share and top-20% share match a negative binomial at that arm's (R, k) to within 0.02. P100ln is the exception (zero 0.44 against 0.51, top 20% 0.69 against 0.73), consistent with its early R being 1.54 rather than the calibrated 1.45.
- **Estimator is not the problem.** I simulated the method-of-moments estimator on negative-binomial samples with known k at the exact infector counts of the runs (11,000 at k 0.3; 3,600 at k 0.9; 12,500 at k 4). The median estimate equals the true value to two decimals in every case, with 5th to 95th percentiles inside plus or minus 10%. So the departures from the composition rule in section 3 are real, not estimator bias.
- **R7 extraction.** The 28 study-level k values give a median of 0.32, an interquartile range of 0.20 to 0.51, a range of 0.04 to 2.97 and 93% below 1, reproducing the review's summary of that paper exactly. The distribution is a far better figure input than the range.

## 2. An error of mine that the full-text checks caught

The `derived_equiv_gamma_k` column I supplied in `16b_published_evidence_starter.csv`, and which has been carried into `17_published_evidence_verified.csv`, is **wrong for transmission shares**. I mapped "19% of cases cause 80% of transmission" to k by a Lorenz curve on the gamma distribution of expected individual infectiousness, which gave k = 0.228 for Adam 2020. The full-text check reports Adam's own estimate as k = 0.43 (0.29 to 0.67) with R = 0.58 (0.45 to 0.72).

My number is wrong because a published transmission share is computed from **realised offspring counts**, which carry Poisson noise on top of the individual variation. That noise concentrates the realised share more than the underlying nu distribution, so for a given share the implied k is larger. Simulating realised negative-binomial offspring instead:

| k | R | realised top fraction causing 80% |
|---:|---:|---:|
| 0.43 | 0.58 | 19.1% |
| 0.29 | 0.45 | 14.8% |
| 0.67 | 0.72 | 24.3% |

Those reproduce Adam's reported 19% (15 to 24%) essentially exactly. The expected-value mapping I used is biased low by about a factor of two here.

Consequences:

1. **Delete or recompute the derived column for E03 and E04.** The correct mapping needs R as well as the share, and R is not in the Hunan abstract (E04), so that row cannot be converted without the full text. For E03 use Adam's own k = 0.43 (0.29 to 0.67) and drop the derived value.
2. **The mapping is still correct for E17 and E18.** Exhaled aerosol and circulating virions are continuous quantities with no counting process, so the Lorenz mapping on expected values is the right one there. This sharpens, rather than removes, the caution I attached to that juxtaposition: emission concentration (equivalent k about 0.21) and transmission concentration (Adam's k 0.43) are now further apart than the raw shares suggested, which weakens any reading that emission heterogeneity alone explains transmission heterogeneity.
3. **Any figure using the derived column needs regenerating.** F4 was not drawn, so only the evidence CSV is affected.

I introduced this in the hand-off; it is not an error by the PanSim agent.

## 3. The most important new result, which the note under-reads

PS1 reports H2 as a failure: the coupled arms land below the pre-registered band. The deeper pattern is visible once the measured k values are converted to variance. For realised offspring the moment k is exactly 1 / CV2 of expected individual offspring, so arms can be compared on a variance scale. Using this run's own A0 (k 4.66, CV2 0.215) as the residual, and the size-biased trait variance among infected cases (shape a + rho for a gamma trait):

| arm | trait CV2 among infected | total CV2 observed | implied residual CV2 | A0 residual |
|---|---:|---:|---:|---:|
| E (episode, a = 0.3) | 3.33 | 3.22 | -0.03 | 0.215 |
| P0 (persistent, rho = 0) | 3.33 | 3.39 | 0.01 | 0.215 |
| P25 | 1.82 | 2.12 | 0.11 | 0.215 |
| P50 | 1.25 | 1.58 | 0.15 | 0.215 |
| P100 | 0.77 | 1.05 | 0.16 | 0.215 |
| P25k (a = 0.08, rho = 0.25) | 3.03 | 3.70 | 0.17 | 0.215 |

If the multiplier and the model's own heterogeneity were independent multiplicative factors, every residual would equal 0.215. Instead the residual is **near zero whenever the trait variance is large**, and recovers toward 0.215 as the trait variance shrinks. The model's intrinsic heterogeneity is not added to the multiplier: it is progressively absorbed by it.

That is a mechanism worth naming, and the most likely candidate is a ceiling on offspring. An agent with a very large multiplier cannot infect more people than it actually meets, so the top of the nu distribution is truncated by the contact structure, and the variance that the schedule would otherwise contribute is spent. Two consequences:

- The "composition rule plus 20 to 50%" heuristic from 09 is not a property of the model. It is what the absorption looks like at one particular trait variance, and it does not transfer, which is exactly what H2 found. The rule should be retired rather than re-fitted.
- **PanSim has an upper limit on expressible individual heterogeneity that is set by the contact structure, not by the multiplier.** This is the person-level counterpart of the kernel result from phase 2 and belongs in the paper as such.

This is an inference from six summary rows, not a measurement. It is directly testable and cheap: see D1 in section 7.

## 4. PS1, what is established

- **Persistence alone is inert.** Arms E and P0 agree on early k (0.311 against 0.295), final size (48,274 against 50,684), zero share, top-20% share and every cohort window. Whether the multiplier is stored for life or redrawn each episode makes no difference to any epidemic observable here. For the paper this is the cleanest available statement that "superspreaders as a persistent type" and "superspreading as a per-episode state" are **observationally equivalent in this model** unless the trait is coupled to exposure.
- **Coupling is what matters, and it has a floor.** Early k rises 0.30, 0.47, 0.63, 0.95 as rho goes 0, 0.25, 0.5, 1, while final size falls 50,684 to 12,680. The size-bias floor predicted by the toy is confirmed in PanSim: at rho 0.5 and rho 1, even a trait as wide as shape 0.05 gives early k 0.46 and 0.83, never approaching 0.33. The observed values sit 10 to 15% below the pure `a + rho` prediction, consistent with the absorption in section 3.
- **Cohort k is not a signature of persistence.** Every main-arm ratio interval includes 1. This closes the question and confirms the revised prior. The mean trait of infected cases is the signature that moves, and only at full coupling does its decline clear replicate noise.
- **A matched-k persistent arm exists but is exotic.** P25k reaches k 0.27 with shape 0.08 and rho 0.25, at the cost of a first-cohort mean trait of 3.4 and a final size of 17,855 against E's 48,274. Matching k does not match the epidemic. That is a second, independent demonstration of the identifiability point already in 03.

One caution for the paper: A0 here gives k 4.66, against 5.25 in phase 2 under the same protocol with different seeds. The intervals overlap (4.09 to 5.47), but the base heterogeneity of this model is only known to about plus or minus 15%, and any composition arithmetic inherits that.

## 5. PS2, a clean null with a caveat

The honest reading is that **PS2 could not test its hypothesis, because this natural history has no peak to speak of**. The 50% window covers days 1 to 19 and holds 95% of the infectiousness area, so "peak mode" is almost the constant multiplier, and f_pre is 0.23, so most transmission happens after diagnosis would occur. The predictions were conditioned on f_pre in advance, which is why Q2 and Q3 failed as expected rather than being reinterpreted after the fact. That is good practice and the design deserves credit, but the conclusion is narrow: for *this* infectiousness profile, a constant multiplier is sufficient for lever studies. It is not a general result about time-varying infectiousness.

Two details worth carrying:

- The 75% window (days 3 to 16, 77% of area) is the only arm where the correlation between an infector's offspring and the generation interval of its offspring is large, 0.38 against 0.10 to 0.11 elsewhere. That is the signature Q4 was looking for, showing up only when the window is genuinely narrower than the infectious period. It suggests the experiment would be informative with a sharper profile, not that the question is closed.
- Pk50's eight-replicate early k is 0.263 against C's 0.297, an 11% gap that the two-seed calibration did not have. The lever comparison is matched on R, less tightly on k.

I would also flag that the profile is truncated at day 21 and about 5% of the recorded area lies beyond day 19, so f_peak for the 50% window is mildly sensitive to where the profile is cut. It does not change any conclusion.

## 6. The evidence tables, and one substantive catch

`17_published_k_from_R7.csv` and `17_published_k_by_setting_R6.csv` are both usable, and the L4 status note is appropriately explicit about the five rows that could not be verified.

**The important catch is in R6, and it changes how we have been citing it.** The file has two tables. The cluster-ascertainment table is the one behind the abstract and behind every use of "household k 1.1, retail k 0.05" in this project, including section 7 of the review and my own comparisons in 10 and 14. The individual-offspring table gives completely different values for the same settings:

| setting | cluster-ascertainment k | individual-offspring k |
|---|---:|---:|
| Households | 1.11 (0.94 to 1.28) | 126 (18 to 280) |
| Restaurants | 0.52 (0.36 to 0.68) | 1.35 (0.92 to 3.40) |
| Care homes | 0.09 (0.03 to 0.16) | 0.27 (0.19 to 0.41) |
| Close-social indoor | 0.10 (0.05 to 0.16) | 0.17 (0.12 to 0.47) |
| Retail and leisure | 0.05 (0.01 to 0.09) | 0.10 (0.05 to 0.35) |

The quantity PanSim measures is individual offspring. So for three phases we have been comparing simulated individual-offspring k against a cluster-level statistic. The note flags the discrepancy; I do not think it follows it through. Two consequences:

1. **The comparison in the paper must use the individual-offspring column**, and on that column the published setting-specific values are 0.10 to 1.35 outside households, which is much closer to what PanSim produces (venue contrasts reached 1.8 to 3.5) than the cluster figures suggested. The gap between model and literature on setting-driven dispersion is smaller than phases 1 to 3 concluded. This should be stated as a correction, not quietly adjusted.
2. **The household value of 126 is informative in its own right.** It says individual offspring within households are essentially Poisson, which supports the reading in 07 that household k near 1 is a property of household size truncation rather than evidence against individual heterogeneity.

## 7. Directions forward

Recommended order. D1 and D2 are small and change what the paper can claim; D3 is the writing; D4 to D6 are optional depth.

**D1. Test the absorption mechanism in section 3 (small, high value).** From traces already on disk, for each PS1 arm: (a) the realised CV2 of the per-episode multiplier among measured infectors, to confirm it is the drawn 3.33 and not compressed by selection; (b) the distribution of distinct contacts per infectious episode, and the correlation between an infector's multiplier and its offspring, which should saturate at high multiplier if the ceiling explanation holds; (c) the offspring distribution of the top 1% of multipliers against a negative-binomial expectation. If truncation is confirmed, the paper gains a clean statement about the model's ceiling on individual heterogeneity. If not, the absorption needs another explanation and the composition arithmetic in the paper should be dropped rather than explained.

**D2. Fix and finish the evidence layer (small, blocking for figures).** Recompute or delete the derived-k column per section 2; replace E03 with Adam's own k; re-cut every setting-specific comparison onto the individual-offspring column of R6 per section 6; add the R7 distribution (median 0.32, interquartile range 0.20 to 0.51) as the headline literature summary in place of the bare range. Then draw F4, the concentration figure, which is the one figure that makes the realised-against-expected distinction visible, and which should now plot transmission shares and emission shares on explicitly separate axes.

**D3. Draft the paper (the main line of work).** The evidence now supports a sharper claim than the outline in 15 assumed. Suggested framing: *dispersion is not decomposition*. Four results carry it, and all four are now in hand:

1. A published k does not identify the mechanism: persistent and per-episode heterogeneity are observationally equivalent (PS1, H1).
2. Matching k does not match the epidemic: P25k reproduces k 0.27 with a third of E's final size.
3. Matching k does not match tail risk: the phase 3 chain result, where the lower-k arm produced no long chains.
4. Matching k does not predict the best intervention: the lever reversal tracks where transmission happens, not k.

Sections 2, 3 and 4 of the outline can be drafted now. I can start on the evidence review and methods sections on request.

**D4. One sharper-profile PS2 arm (optional).** The 75% window result suggests the question is live for a pathogen with a narrow infectious window. A single arm with an artificially compressed profile would show whether the lever ranking is stable in general or only for this natural history. Worth doing only if the paper wants to claim generality.

**D5. The lever predictor (optional, currently exploratory).** The household-share rule from 14 section 6 stays a hypothesis with n = 5 until arms are run at shares around 0.25, 0.37 and 0.45 with the share measured in the same runs.

**D6. Venue work stays parked** until the national and regional rebuild lands, as decided. The finding in section 6 slightly lowers its priority for the paper, because the model-literature gap on setting dispersion is narrower than we thought.

## 8. What I would not do

Re-fit the composition rule to the coupled arms. It is a description of one regime, and section 3 suggests it has no mechanism behind it.

Report P100ln's final size against the gamma arms without the caveat that its twelve-replicate early R is 1.54 rather than the calibrated 1.45.

Quote the 19% to 80% share and a k value from the same study in the same sentence without stating which is realised and which is expected. Section 2 is what happens when those are mixed.

## 9. Method note

Checks run: recomputation of peak, both windows, both f_peak values and f_pre from the raw profile days; negative-binomial simulation of zero and top-20% shares for all eight PS1 arms; method-of-moments estimator bias at the runs' own infector counts (400 repetitions per setting); the variance decomposition in section 3; realised-offspring top-share simulation at Adam 2020's reported k and R; summary statistics of the R7 and R6 extractions. The absorption mechanism in section 3 is an inference from summary values and is not confirmed. No access to the raw traces or the figure images.
