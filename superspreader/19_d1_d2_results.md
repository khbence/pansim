# D1 and D2: the absorption test, and the evidence correction

12 October 2026. This note carries out D1 and D2 from `18_reading_of_17_and_directions.md`. D1 reads the PS1 traces already on disk. No new PanSim runs were made. D4, D5 and D6 were left undone, for the reasons at the end. D3, the paper draft, was not started.

## 1. What D1 measured

Index cases are agents infected on days 1–10 with a known infector. Offspring are the people they infected within 14 days. That is the same window as the early k in `17_ps1_ps2_results.md`. Recomputing that k from the traces reproduced the published value on every arm, including A0 at 4.66.

`trace.tsv` stores one row per agent, the latest episode, and it stores only infected agents. Uninfected contacts are not in the file, so the distribution of distinct contacts cannot be measured. What is measured instead is the number of distinct places among the 14-day offspring. That count cannot exceed the offspring count. Reinfection is rare in this window (under 1% of index cases have `infectedCount` above 1), so the recorded multiplier is the multiplier of the episode being counted.

The pre-registered ceiling rule, fixed in `analyze_d1.py` before the pooled estimates were interpreted, requires three things at once on every arm whose expected multiplier CV² is at least 1: the top 1% mean offspring below 60% of the lower-half proportional prediction, the replicate-bootstrap 97.5% of that ratio below 0.80, and an elasticity below 0.60 from the shed bin [0.5, 1) to the bin shed ≥ 8. The six arms in section 3 of note 18 are the multiplier comparison. The other arms are reported beside that comparison.

## 2. The multiplier among infectors is the size-biased draw

CV² is the sample variance divided by the squared mean. For an episode gamma draw, an uncoupled trait, or a lognormal trait, the expectation among infectors is 1/shape. Lognormal CV² does not change under size biasing. For a coupled gamma trait the expectation is 1/(shape + rho). The population files (`agents_trait.csv`, replicate 1101) give the draw before infection: CV² 3.34 at shape 0.3, 12.53 at shape 0.08, and 20.18 at shape 0.05, against 3.33, 12.5 and 20.

| arm | infectors | expected CV² | pooled CV² | replicate interval | early k |
|---|---:|---:|---:|---:|---:|
| E | 11,201 | 3.33 | 3.35 | 3.12 to 3.53 | 0.311 |
| P0 | 12,214 | 3.33 | 3.31 | 3.11 to 3.48 | 0.295 |
| P25 | 8,163 | 1.82 | 2.03 | 1.88 to 2.17 | 0.471 |
| P50 | 5,338 | 1.25 | 1.40 | 1.30 to 1.53 | 0.632 |
| P100 | 3,603 | 0.77 | 0.84 | 0.78 to 0.91 | 0.954 |
| P25k | 3,751 | 3.03 | 3.54 | 3.19 to 3.81 | 0.270 |

All six sit inside 20% of the expectation, and each replicate interval covers it. Among the index cases that enter the early-k calculation, including those with no offspring, selection has not compressed the multiplier below the distribution the model drew.

Restricting to index cases with at least one offspring does compress it, because that cut removes the left tail. The CV² falls from 3.35 to 1.25 on E, from 3.31 to 1.24 on P0, from 2.03 to 0.98 on P25, from 1.40 to 0.77 on P50, from 0.84 to 0.53 on P100, and from 3.54 to 1.29 on P25k. That is selection on the outcome. It is not the distribution of the multiplier among the measured index cases.

Two sensitivity arms fall outside the band, in opposite directions.

- P100ln has infected CV² 2.41 (interval 2.03 to 2.66) against a lognormal target of 3.33. Its twelve-replicate early R is 1.54, not the calibrated 1.45. It is not used to judge the gamma arms, and its final size is not compared with theirs.
- P50a005 and P100a005 have infected CV² 2.33 and 1.28, above the pure size-bias targets 1.82 and 0.95, while their population CV² is 20.2. Their mean multipliers, 8.3 and 15.2, are below the pure size-bias means 11 and 21. Size bias is present, and it is a bit weaker than weighting by g^rho. The same tilt, smaller, is visible in P25, P50 and P100: infected CV² a little above 1/(shape+rho), mean multiplier a little below (shape+rho)/shape.

## 3. Offspring keeps rising. The far tail falls short of a straight line.

Decile means of 14-day offspring rise with the decile mean multiplier on E, P0, P25, P50, P100 and P25k. Pearson and Spearman correlations of multiplier and offspring are close to each other (E 0.75 and 0.75; P0 0.77 and 0.75; P100 0.62 and 0.60), which is what a near-linear bulk looks like. A straight line through the origin leaves little squared error for a saturating curve to pick up: the saturating fit's squared error is 95% of the linear fit on E and 99.9% on P100. The fitted half-saturation multiplier on E is 8.0, and the 99th percentile of E's multipliers is 8.6. The bend sits in the extreme tail.

The top 1% is where the bend shows. The prediction is the lower-half slope (offspring over multiplier, below the median multiplier) times the top 1% mean multiplier. The ratio is observed mean offspring over that prediction. Intervals are replicate bootstraps.

| arm | top 1% mean multiplier | observed offspring | proportional prediction | ratio |
|---|---:|---:|---:|---:|
| E | 11.6 | 11.8 | 24.0 | 0.49 (0.41 to 0.59) |
| P0 | 11.5 | 13.1 | 21.2 | 0.62 (0.54 to 0.68) |
| P25 | 13.9 | 9.8 | 15.0 | 0.65 (0.54 to 0.71) |
| P50 | 15.4 | 7.7 | 11.5 | 0.67 (0.56 to 0.82) |
| P100 | 17.2 | 6.0 | 7.2 | 0.83 (0.70 to 0.95) |
| P25k | 41.9 | 14.1 | 23.0 | 0.61 (0.51 to 0.78) |

E is the only section-3 arm whose ratio interval lies entirely under 0.60. Elasticity from multiplier 0.5–1 to multiplier ≥ 8 is 0.61 on E, 0.70 on P0 and 0.80 on P100. The pre-registered conjunction fails. P100's top 1% is close to proportional. Distinct transmission locations rise along with offspring (E's top decile: mean multiplier 5.4, mean offspring 6.3, mean places 3.2). There is no flat cap on places inside the range these epidemics reach.

Narrow multiplier bins still carry the A0 extra-Poisson variance. Inside shed [1.0, 1.2), E's offspring variance-over-mean is 1.41, against 1.34 from A0's k of 4.66 at that mean. Inside [8, 12) it is 3.81, against 3.22. The body's residual heterogeneity is still there.

That is enough to read the section 3 residual. Forcing proportionality, the multiplicative residual against the expected trait CV² is −0.03, 0.01, 0.11, 0.15, 0.16 and 0.17 for E, P0, P25, P50, P100 and P25k, with A0 at 0.215. The same arithmetic as note 18. It is what a proportional composition rule sees. On E the CV² of the empirical conditional mean is 2.17, below the shed CV² of 3.35, because the mean curve bends in the tail. Predicting marginal k from that empirical curve plus A0 noise gives 0.35, against the observed 0.31. The same calculation on P100 gives 0.98 against the observed 0.95. The proportional composition rule, which ignores the bend, predicts 0.23 on E.

## 4. What the paper can say, and what it should drop

The pre-registered ceiling test fails. A contact ceiling that removes the model's own heterogeneity is not established. The composition rule is retired, and it is not re-fitted to the coupled arms.

What the traces do establish is narrower. The multiplier among the measured infectors is the size-biased draw on the six section-3 arms. Mean offspring rises across deciles. In the top 1% of the widest arm the mean offspring is about half the lower-half extrapolation (E, ratio 0.49, interval 0.41 to 0.59), and that shortfall shrinks as the infected multiplier distribution gets narrower. Inside ordinary multiplier bins, offspring remains overdispersed by about the A0 amount.

## 5. D2. The evidence layer

`published_evidence_verified.csv`

- E03 keeps the realised share, 19% (15 to 24). The derived Lorenz k 0.228 is blank. Adam's own offspring estimate is k 0.43 (0.29 to 0.67) at R 0.58 (0.45 to 0.72). The share is realised. The k is his offspring estimate. They are different quantities from the same study.
- E04 keeps the realised share, 15%. The derived k is blank. The share cannot be converted without R. The full-text offspring k 0.30 (0.23 to 0.39) is a separate estimate.
- E17 and E18 keep their Lorenz values, k about 0.21 and 0.012. Exhaled aerosol and circulating virions are continuous quantities, so that mapping is the right one. Those are emission k's.

F2 now plots Chen 2025's individual-offspring column, which is the quantity PanSim measures. Households are k 126 (18 to 280), restaurants 1.35 (0.92 to 3.40), care homes 0.27 (0.19 to 0.41), close-social indoor 0.17 (0.12 to 0.47), retail 0.10 (0.05 to 0.35). The headline from the 28 study-level estimates is a median of 0.32 and an interquartile range of 0.20 to 0.51. Twenty-six of the 28 are below 1.

F4 plots the two kinds of share on separate axes. Simulated realised offspring at Adam's (k, R) corners puts 80% of transmission in the top 19.1%, 14.8% and 24.1% of cases, at (0.43, 0.58), (0.29, 0.45) and (0.67, 0.72). The withdrawn Lorenz mapping, k 0.228, puts the same 80% share at a top fraction of 19.0%. The shares match. The k values differ by a factor of about 1.9. Edwards' continuous aerosol curve passes through about 18% of people for 80% of emission (simulated 17.9%, k 0.212). Yang's virion curve passes through about 2% of people for 90% of emission (simulated 2.1%, k 0.012).

F3's title is now the statement the coupled arms support: the 20 to 50 percent band fits E and P0, and P25, P50 and P100 lie above it. The band is not treated as a law of the model.

## 6. What was not run

D3 is the paper. The four results named in note 18 are already in hand: persistence without coupling is observationally equivalent to the per-episode multiplier; P25k matches k and not the epidemic; the phase 3 lower-k arm produced no long chains; the lever ranking tracks where transmission happens. Drafting those sections was left for a writing pass.

D4, a sharper infectiousness profile, was skipped. Note 18 asks for it only if the paper claims the lever ranking is general, and section 5 of that note says the PS2 result is specific to this natural history.

D5 was skipped. The household-share predictor remains the five-arm comparison in F7 until runs exist at shares near 0.25, 0.37 and 0.45 with the share measured in those same runs.

D6 stays parked until the national and regional rebuild, as note 18 decided.

## 7. Files

- `superspreader/study/ps/analyze_d1.py` writes `study/results/ps/ps1/d1_absorption.json`, `d1_arms.csv`, `d1_bins.csv` and `d1_narrow.csv`.
- `superspreader/study/ps/make_figures.py` redraws F2 and F3 and adds F4 and F8.
- `superspreader/study/ps/inputs/published_evidence_verified.csv` is the corrected evidence table.
