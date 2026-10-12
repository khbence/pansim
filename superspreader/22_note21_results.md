# Note 21 experiments: ward-split chains, matched-size levers, Nipah calibration, and the three analyses

12 October 2026. This runs the items in `21_further_experiments.md`. X4 was not run: the note says to run a sharper infectiousness profile only if the paper claims the lever ranking is general, and it says the draft does not. X6 was not run: it needs the simulator to emit co-present people per episode, and the note treats that as an output change to request only if the paper wants to explain the tail bend.

No claim below re-fits the composition rule.

## X1. The heavy chain tail does not survive ward subdivision

Hospital (type 12), health-centre (type 14) and nursing-home (type 22) locations were split into wards of about 30 assigned agents. All 120 of those locations were large enough to split, into 12,886 wards. Assigned agents per ward run from 28 to 34. Total capacity is unchanged (106,990 before and after). Each agent keeps every schedule slot, moved onto one ward of the original place, so only co-presence changes. The two phase-3 chain arms were recalibrated to an all-node mean offspring of 0.33 and run for 40 replicates (about 38,500 chains each).

| arm | chains | mean offspring | k | chains > 10 | NB expected > 10 | chains > 50 | NB expected > 50 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat, shape 0.04, wards | 38,534 | 0.341 | 0.043 | 525 | 540 | 19 | 30 | 138 |
| caregiving, shape 0.09, wards | 38,520 | 0.317 | 0.151 | 195 | 301 | 0 | 1.0 | 42 |
| flat, shape 0.04, original aggregate | 38,630 | 0.294 | 0.043 | 416 | 437 | 17 | 14 | 101 |
| caregiving, shape 0.09, original aggregate | 38,658 | 0.320 | 0.083 | 236 | 423 | 19 | 4.4 | 249 |

The flat arm still follows its negative binomial. The caregiving arm's heavy extreme does not. On the aggregated map it produced 19 chains above 50 against 4 expected, and a maximum of 249. On the ward map it produced none above 50, against 1 expected, and a maximum of 42. The thin middle is smaller than it was but has not gone: 195 chains above 10 against 301 expected, where the aggregate arm was 236 against 423.

The hospital-to-hospital correlation that carried the original claim collapses with the aggregation. In the aggregate caregiving arm, a case infected in hospital had mean offspring 0.68 and 70% of its children were infected in a hospital (3,813 such cases). In the ward arm those figures are 0.33, 16%, and 527 cases. Same-location transmission among all child links is 19% in the flat ward arm and 39% in the caregiving ward arm, so caregiving still concentrates some transmission inside a place, without producing the extreme chains.

Section 5.7 cannot stand as a transmission-structure result independent of venue aggregation. The heavy tail was an artefact of hospitals and health centres that held hundreds of people at once. What remains is a milder deficit of mid-sized chains.

## X5. The lognormal floor survives matched growth

The full-coupling lognormal arm was recalibrated to P100's twelve-replicate early R of 1.337. Two-seed calibration landed at 1.29. The twelve-replicate early R is 1.303 (pooled 1.307), against P100's 1.337. Pooled early k is 0.479, against 0.464 in the old arm whose early R was 1.54, and against P100's gamma k of 0.954. Matching growth does not close the gap. The dispersion floor under coupling remains specific to the gamma family in this comparison.

## X2. Matched final size only occurs once both epidemics are large

Set I (shape-1 multiplier, venues off) and Set II (covid venue profile, strength 8) were evaluated at five coefficients each. Near the original early R of about 1.2, final sizes are 51,906 and 20,728. The two (early R, final size) curves meet only at the top of the grid: early R 2.05 and 2.01, final sizes 155,051 and 143,554. One coefficient cannot match both early R near 1.3 and final size. The lever comparison at that meeting point, four replicates, reductions against the uncontrolled arm:

| lever | Set I, R 2.05, size 155,502 | Set II, R 2.01, size 143,503 |
|---|---:|---:|
| masks | 17.8% | 32.9% |
| close venue types | 10.2% | 54.6% |
| isolate diagnosed + household | 8.1% | 10.7% |
| isolate diagnosed only | 0.2% | 0.5% |
| household share of infections, no lever | 0.383 | 0.149 |

Venue closure still beats household isolation in the setting arm, by 55% to 11%. In the person arm the original ranking, household isolation well above venue closure, does not survive: the two reductions are 8% and 10%. Day-21 levers are weak on both arms once the uncontrolled epidemic has already reached about 150,000 infections. The saturation objection is therefore only half closed. The setting arm's preference for venue closure is not an artefact of Set II being the smaller epidemic. The person arm's preference for household isolation was tied to the smaller, slower epidemic and is not there once the two final sizes match.

## X2. The household-share sign change sits between 0.37 and 0.39

The prediction was fixed before these lever runs: venue closure reduces more infections than household isolation when the household share is below 0.37, and household isolation reduces more above 0.37. Three mechanisms were chosen from a six-arm pilot because their uncontrolled household shares landed nearest 0.25, 0.37 and 0.45. The nearest the pilot came to 0.45 was 0.389. Shares below are the four-replicate means in the same runs as the levers.

| arm | household share | venue closure | household isolation | venue minus household |
|---|---:|---:|---:|---:|
| covid venues, strength 3 | 0.246 | 73.9% | 43.0% | +30.9 points |
| covid venues, strength 1 | 0.368 | 65.4% | 45.8% | +19.6 points |
| multiplier shape 0.5, venues off | 0.389 | 44.3% | 55.2% | −10.9 points |

At 0.368, just under the stated threshold, venue closure still wins. At 0.389 it has lost. The sign change is real and it sits between 0.37 and 0.39. The gap narrows as the household share rises: +31 points, then +20, then −11. Masks stay near 76% to 79% on all three arms, so the movement is between the two targeted levers. The 0.45 share was not reached by any pilot arm; the shape-0.5 arm at 0.39 is the highest new point.

## X3. A flat multiplier of shape 0.03 meets the Nipah-like marginal targets, and the chain tail follows

The literature joint target, with the caveat that R and the zero share come from different studies, is mean offspring 0.33, about 92% of cases infecting nobody, and dispersion near 0.036. That pair implies roughly a 1% chance that five seeds produce more than 50 cases. A negative-binomial branching process at (0.33, 0.036) gives 0.61% in 20,000 simulated five-seed outbreaks.

Shapes 0.02, 0.025, 0.03, 0.035 and the existing shape 0.04 were compared after calibrating the coefficient to mean offspring 0.33. The shape closest to k = 0.036 and the shape closest to a 92% zero share are the same shape, 0.03 (probe k 0.037, zero share 0.924). There is not a second arm. Forty replicates of shape 0.03:

| | shape 0.03, 40 replicates | negative binomial at the realised mean and k |
|---|---:|---:|
| chains | 38,545 | |
| mean offspring | 0.363 | |
| k | 0.034 | |
| zero share | 0.919 | |
| chains > 10 | 540 | 610 |
| chains > 50 | 61 | 60 |
| maximum | 254 | |
| five-seed outbreaks above 50 cases | 0.77% | 0.95% |

The full-sample mean drifted up from the two-seed calibration (0.318) to 0.363. Dispersion and the zero share land on the Nipah-like targets. Once they do, the chain-size tail matches the negative binomial: 61 chains above 50 against 60 expected. Calibrating to k and calibrating to the zero share are the same choice of multiplier in this flat model, and the chain-size distribution does not then pull away from that negative binomial. The divergence between a marginal k and the chain tail remains the caregiving result in X1 and in phase 3, which is a property of where transmission happens, not a failure of the flat multiplier to hit a Nipah-like (k, zero share) pair.

## A1. Equivalence is within about 11% on early k, and final sizes are not separated

Paired across the twelve shared seeds, P0 minus E in early k is −0.011 (95% interval −0.045 to +0.023). The half-width is 11% of E's early k. Reaching ±5% would take about 56 replicates at the same variance, which was not run. Final size is 48,274 for E (interval 45,744 to 50,803) and 50,684 for P0 (47,886 to 53,482). The paired difference is +2,410 (interval −1,547 to +6,367). The paper can call the arms equivalent inside those bounds. It should not call them exactly equivalent.

## A2. Seventeen against nineteen is not a difference

Chains above 50 are 17 of 38,630 in the flat shape-0.04 arm and 19 of 38,658 in the caregiving arm. A two-proportion test gives p = 0.74. Detecting a doubling of that rate at 80% power would take about 51,000 chains per arm. The caregiving claim rests on the shape of the tail: 235 chains above 10 against 423 expected from a negative binomial with that arm's own mean and k, while the flat arm is 416 against 437. Those expected counts were resimulated here (300,000 chains) and match the phase-3 account.

## A3. A hundred index cases do not separate k of 0.2 from k of 0.5

Four thousand simulated contact-tracing studies, method of moments, true k = 0.3, reproduction number 1. The 5th to 95th percentiles of the estimate are 0.19 to 0.56 from 100 index cases, 0.21 to 0.47 from 200, and 0.23 to 0.40 from 500. At 100 index cases the interval covers both 0.2 and 0.5. That is a property of the estimator and the sample size, not of PanSim.

## Files

- `study/build_wards.py` writes `study/inputs/x21/ward_summary.json`. The rewritten agent and location files are local inputs and are not committed.
- `study/run_x21.py` ran X1, X5, X2 and X3. Raw traces stay under `study/results/x21/` and are gitignored.
- `study/analyze_note21.py` writes `study/results/x21/analyses.json` for A1–A3 and the ward-chain summary.
- Chosen coefficients and the household-share pilot are in `study/results/x21/x2/match.json` and `pilot.json`. The shape grid is in `study/results/x21/x3/grid.json`.
