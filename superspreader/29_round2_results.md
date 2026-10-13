# Note 28 experiments

13 October 2026. Protocol: `28_pansim_agent_instructions_round2.md`. The superseded drafts were not used. Thresholds written as "above" mean size strictly greater than the cap. Negative-binomial expectations are the exact Dwass total-progeny probabilities at the arm's own mean and k. Early R in a ratio is measured on the same replicates as the numerator.

Maps: V4, V5, V3 and the open-epidemic arms use `locations1.json` and `agents2.json`, except where a virtual split is named. W4b uses `locations0.json` and `agents1.json`. W10 step 5 uses `locations3.json` and `agents4.json`.

## Predictions

| prediction | result |
|---|---|
| W1 caregiving empirical-iid lands nearer 220 than 313 | failed: 293, nearer the negative binomial |
| W1 flat empirical-iid, negative binomial and observed agree | met: 487, 496 and 485 above 10 |
| W1, a study of 38,613 chains reaches 585 under the empirical offspring distribution in more than 5% of studies | not the outcome: 0.33% of studies |
| W2 cinemas are at least 25% of infections in chains above 200, and one cinema holds at least 30% of the 585 chain | failed |
| W4a reference-free ratio at least 0.80 at shapes 0.3, 0.08 and 0.03 | met on the point estimates (0.91, 0.87, 0.86). The intervals for 0.08 and 0.03 extend slightly below 0.80 |
| W4a a ceiling needs the ratio below 0.75 and the upper bound below 0.85 | not met at any shape |
| W3 longest chain at most 300 and chains above 100 inside the Poisson band | met: longest 202, 2 against 2.1 |
| W4b old shape 0.3 ratio about 0.76, and old 0.08 and 0.03 within 0.1 of the new map | met |
| W4b old ratios clearly lower at every shape, intervals not overlapping | not the outcome |
| W5 at matched early R, E and P0 final sizes within 10% and early k within 13% | k met (9%). Final size failed: 14.5%, short of the 15% bar |
| W6 restaurant median fill 0.5 to 1.0 and 90th percentile above 2 | met: 0.93 and 3.25 |
| W7(a) at (0.3%, 1000x) early k between 0.04 and 0.15, median cluster at most 100 | k failed (0.36). Median met (0) |
| W7(b) the largest clusters come from cinemas or churches | met: cinemas |
| W8 chains above 10 move to within 15% of the negative binomial | met: 8.5% below |
| W10 coefficient still accepts at early R 1.39 | met: 0.00041, two-seed R 1.375 |
| W10 restaurant share within 0.5 points of 4.28% | met: 4.27% |
| W10 peak prevalence moves by less than 2% | failed: 5.4% lower |
| W10 longest chain falls below the W3 value of 202, and below 250 | met: 147 |
| W10 at most 1% of restaurants peak above capacity | failed: 83%, at a target fill of 1.20 |
| W10 fill-pinned and full-volume early k differ by more than their replicate ranges | met: 3.34 against 1.54 |
| W9 sign of venue closure minus household isolation changes between shares 0.37 and 0.39 | positive at 0.365 (+21 points) and negative at 0.424 (−11 points). No lever arm landed inside 0.37 to 0.39 |
| W9 closure benefit at a matched share moves by less than 5 points from X2 | met: 1.9 and 1.7 points at the two matched shares |
| W9 a household share above 0.39 | met: 0.424 |

## W1. Empirical offspring, drawn independently

Each infected agent in the trace contributes one offspring count, including zeros. A branching process then draws from that list. V4 used 300 blocks of 38,613 chains (11.6 million chains). The other arms used 100,000 chains. Counts below are scaled to the arm's own number of chains. The Poisson band is the 95% Garwood interval of the observed count.

| arm | source | above 10 | above 50 | above 100 | above 200 | longest |
|---|---|---:|---:|---:|---:|---:|
| V4 shape 0.03 | observed | 642 (593–694) | 71 | 15 | 4 | 585 |
| V4 | empirical iid | 657 | 75 | 18 | 1.8 | |
| V4 | negative binomial | 669 | 84 | 17 | 1.4 | |
| V5 flat | observed | 485 (443–530) | 19 | 0 | 0 | 99 |
| V5 flat | empirical iid | 487 | 17 | 0.4 | 0 | |
| V5 flat | negative binomial | 496 | 17 | 0.8 | 0 | |
| V5 caregiving | observed | 220 (192–251) | 0 | 0 | 0 | 44 |
| V5 caregiving | empirical iid | 293 | 0 | 0 | 0 | |
| V5 caregiving | negative binomial | 315 | 0.6 | 0 | 0 | |
| X1 caregiving | observed | 195 | 0 | 0 | 0 | 42 |
| X1 caregiving | empirical iid | 291 | 0 | 0 | 0 | |
| X1 caregiving | negative binomial | 301 | 0.5 | 0 | 0 | |

The flat arm agrees. The caregiving arms do not: 293 sits outside the band around 220 and next to the negative binomial. Independent draws from the caregiving offspring distribution do not produce the thin middle. The offspring are correlated across generations.

For the 585 chain, 1 of 300 studies of 38,613 empirical-iid chains contained a chain of at least 585 (share 0.0033). The negative binomial gives 0.00048 for the same event, and 0.062 for a chain of at least 254 on the old map at its own parameters. The offspring tail makes a chain of 585 more common than the negative binomial does, and still uncommon. The 585 chain is not explained by the shape of the offspring distribution alone.

## W2. What the long chains are made of

On the new map, 15 chains are above 100, 4 are above 200, and 1 is above 500. The longest is 585 cases in 9 generations, replicate rep09.

| | restaurant | church | park | cinema | shop | hospital | home |
|---|---:|---:|---:|---:|---:|---:|---:|
| the 585 chain | 1.9% | 0.9% | 3.1% | 6.7% | 16% | 56% | 13% |
| four chains above 200 | 4.7% | 0.9% | 5.4% | 6.2% | 24% | 40% | 13% |
| 67 chains of 51 to 200 | | 2.1% | | 7.1% | | 10% | |

One hospital, location `64118_1496` (type 12, capacity 750, index 75375), holds 55% of the 585 chain. The next chain, 229 cases, is 53% hospital, and location `64143_1498` holds 52% of it. The cinema share is 6.7%, against 4.6% of infections overall in V1. No cinema, church or park holds 30% of the 585 chain.

The old-map shape-0.03 arm has 14 chains above 100, 1 above 200 and none above 500. Its longest chain is 254. The largest single location in that chain holds 15%, and the hospital share is 15%. The 585 chain is a hospital chain. The old map's longest chain was not.

## W4a and W4b. Reference-free offspring curve

Index cases are infections on days 1 to 10 with a known infector. Offspring are counted within 14 days. The ratio is the top 1% mean offspring divided by early R times the top 1% mean multiplier. Intervals resample replicates.

| map | shape | early R | top 1% multiplier | top 1% offspring | ratio | interval |
|---|---:|---:|---:|---:|---:|---|
| new | 0.3 | 1.341 | 11.63 | 14.14 | 0.91 | 0.81 to 1.00 |
| new | 0.08 | 1.364 | 26.82 | 31.82 | 0.87 | 0.76 to 0.98 |
| new | 0.03 | 1.455 | 44.47 | 55.55 | 0.86 | 0.79 to 0.94 |
| old | 0.3 | 1.333 | 11.58 | 11.83 | 0.77 | 0.67 to 0.87 |
| old | 0.08 | 1.411 | | | 0.78 | 0.70 to 0.91 |
| old | 0.03 | 1.404 | | | 0.91 | 0.80 to 1.04 |

A ceiling claim needs the ratio below 0.75 and the upper bound below 0.85. No shape meets that. The point estimates on the new map are all at least 0.80. The old shape-0.08 and shape-0.03 ratios are within 0.1 of the new ones (0.09 and 0.05). The intervals overlap at every shape, including shape 0.3 (0.67–0.87 against 0.81–1.00). The old map is not clearly lower at every shape.

The original lower-half ratio remains the one in note 26: 0.62 at shape 0.3 on the new map, and unusable at shapes 0.08 and 0.03 because the median multiplier is essentially zero. The binned curve is in `29_offspring_curve.csv`.

## W5. Final size at a matched early reproduction number

A0 was already inside the band and was not re-run: twelve-replicate early R 1.382, final size 58,613, early k 5.23. E and P0 were corrected from the twelve-replicate R, not from the two-seed R. The seed-to-seed standard deviation of early R is 0.073 for E and 0.077 for P0, which is why a two-seed calibration missed by 0.12.

| arm | coefficient | early R | early k | final size |
|---|---:|---:|---:|---:|
| A0 | 0.00041 | 1.382 | 5.23 | 58,613 |
| E | 0.000498 | 1.377 | 0.291 | 55,739 |
| P0 | 0.000511 | 1.388 | 0.317 | 63,832 |

E and P0 differ by 0.011 in early R. Their early k values differ by 9%, inside 13%. Their final sizes differ by 14.5% (63,832 against 55,739). That misses the 10% prediction and does not reach the 15% gap that would retire equivalence for the size of the epidemic. A0 sits between them.

## W3. Cinemas and churches split, on the hospital-ward map

Type 6 and type 9 were split by observed peak, targeting a median peak near 150 for cinemas and 300 for churches. Hospitals, health centres and nursing homes stayed in the wards of about 30 from V5. Shape 0.03, flat multiplier, 40 replicates, calibrated to mean offspring 0.33. The realised mean is 0.311 and k is 0.035.

| | observed | negative binomial |
|---|---:|---:|
| chains | 38,583 | |
| above 10 | 454 | 489 |
| above 50 | 25 | 26.4 |
| above 100 | 2 | 2.1 |
| above 200 | 1 | 0.03 |
| longest | 202 | |

The longest chain is 202, which is at most 300, and the two chains above 100 sit inside the Poisson band of that count (0.2 to 7.2). The five-seed probability of a total above 50 is 0.43% against an exact 0.48%. The numerical branch of the prediction is met.

The 202 chain is 44% shop, 17% home and 8% cinema. Its largest location holds 25%. W2 showed that the 585 chain was a hospital, and this map split hospitals as well as cinemas and churches. The fall from 585 to 202 is what the hospital split, not the cinema split, was in a position to do.

## W6. Fill, assignment, and the hospital hold-out

From the existing three-week occupancy file, peak divided by design capacity:

| type | locations | fill median | fill 90th | fill max | share above capacity |
|---|---:|---:|---:|---:|---:|
| 5 restaurant | 420 | 0.93 | 3.25 | 7.75 | 0.47 |
| 6 cinema | 20 | 1.40 | 2.89 | 4.69 | 1.00 |
| 9 church | 52 | 3.72 | 7.53 | 15.9 | 0.98 |
| 10 park | 167 | 0.39 | 0.74 | 1.74 | 0.02 |
| 12 hospital | 17 | 0.57 | 0.59 | 0.63 | 0 |

The restaurant prediction, median fill between 0.5 and 1.0 and a 90th percentile above 2, was already met. Assignment and capacity are almost uncorrelated for restaurants (correlation 0.03). The busiest peak of 310 is a capacity-40 cafe.

Eight new-map 37-week replicates, with the variant vector left as it was, give a 105-day hospital area of 10,199 (the eight values run from 7,980 to 11,597). The eight-replicate baseline, including the two earlier replicates stored in `p2_waves.json`, is 10,187. The difference is 12. The minimum detectable difference at 80% power is 1,765. Wave-2 peak hospital occupancy is 224 against 227, difference 4, minimum detectable difference 49. Wave 2 does not differ from the baseline by more than that, so the variant vector was not refitted.

## W8. Home weight in the caregiving profile

W1 said the thin middle is correlation, so this arm was run. It is the V5 caregiving arm with the home base factor set to 0.35 instead of 2.5, hospitals and restaurants already split, 40 replicates. Realised mean offspring 0.329, k 0.097.

Chains above 10: 392, against a negative-binomial expectation of 428. That is 8.5% below, inside 15%, and 428 sits inside the Poisson band of 392 (354 to 433). On the unchanged caregiving arm the same comparison was 220 against 315, 30% below. Lowering the home weight removes the thin middle. The longest chain is 61.

## W7. Events

Events are a daily draw. Configuration (a) is eligible only at restaurants and long-visit shops. Configuration (b) also includes cinemas and churches. Each arm was calibrated to early R 1.39 on four seeds and then run for 12 replicates of eight weeks. The k below is the twelve-replicate early window, and it includes infections that happened at events. Cluster size is infections at that location on that day. Most events infect nobody, so the median is zero.

| configuration | rate | factor | early R | early k | p90 cluster | maximum | type of the maximum | share of infections at events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| restaurants and long shops | 0.3% | 1000 | 1.370 | 0.359 | 47 | 615 | long shop | 0.13 |
| restaurants and long shops | 1% | 200 | 1.351 | 0.852 | 15 | 258 | long shop | 0.16 |
| restaurants and long shops | 0.3% | 100 | 1.373 | 3.23 | 12 | 97 | long shop | 0.03 |
| restaurants and long shops | 1% | 20 | 1.368 | 4.41 | 3 | 54 | long shop | 0.02 |
| all social types | 0.3% | 1000 | 1.234 | 0.211 | 30 | 944 | cinema | 0.23 |
| all social types | 1% | 200 | 1.366 | 0.661 | 15 | 565 | cinema | 0.23 |
| all social types | 0.3% | 100 | 1.312 | 2.82 | 10 | 229 | cinema | 0.05 |
| all social types | 1% | 20 | 1.376 | 4.33 | 3 | 89 | cinema | 0.04 |
| restaurants and long shops | 0.3% | 2000 | 1.380 | 0.121 | 53 | 1,796 | long shop | 0.18 |

At (0.3%, 1000x) in configuration (a), early k is 0.36. The registered band was 0.04 to 0.15, a factor of two around the old E5 value of 0.076. The median cluster is 0, which is at most 100. The k prediction failed. On the subdivided restaurant map, a strong event does not reproduce the very wide dispersion E5 reported when restaurants were eight city-wide rooms. That is the outcome the protocol said would mean the old E5 result was produced by those aggregates.

In configuration (b) the largest cluster in every arm is a cinema.

Early k of 0.3 in configuration (a), at a rate of 0.3%, sits between a factor of 1000 (k 0.36) and a factor of 2000 (k 0.12). A log-linear interpolation puts it near a factor of 1,200. At the 1000x point the 90th percentile of cluster size is 47 and the maximum is 615. At 2000x the 90th percentile is 53 and the maximum is 1,796. A Skagit-sized cluster, 53 cases among 61 attendees, is around that 90th percentile. It is not the typical event. The typical event infects nobody.

## W10. Restaurant assignment

Per-venue presence rates were measured on the three-week occupancy file. Among 397 venues with at least 50 assigned agents and five occupied days, the median presence rate is 0.077 (5th percentile 0.062, 95th 0.115). Twenty-three venues used that median because their own rate was unstable. At those rates a target fill of 1.10 absorbs 441,000 of the 461,130 assignments. Fill 1.15 absorbs them with 66 slots of slack, which is not the required 2%. Fill 1.20 absorbs 481,000 after rounding the quota down, which is 4% of slack.

The reassignment is greedy: agents in a fixed random order (seed 20261013), each given its three nearest age-eligible venues that still have quota. `locations3.json` is a copy of `locations1.json`. `agents4.json` changes only type-5 location ids. All 153,710 agents still have three distinct type-5 slots, the assignment total is 461,130, and all 420 venues are used. Median displacement is 33 metres. The 95th percentile is 2.0 km, under the 5 km escalation. Fifty-seven percent of agents changed venue. The raw maximum assignment is 6.4 times the median, because design capacities run from 40 to 200. Implied fill, presence rate times assigned divided by capacity, has median 1.20 and maximum 1.20.

Churches still cannot be balanced. At the measured presence rate their peak if perfectly balanced is about 645, and they need about 170 more locations. None were invented.

The repaired file, eight replicates, keeps the coefficient 0.00041. The two-seed early R is 1.375 and the eight-replicate early R is 1.401, against 1.406 on the previous file. The restaurant share of infections is 4.27%, against 4.28%. Peak prevalence is 12,791 against 13,526, a fall of 5.4%. The eight old peaks run from 13,016 to 13,973 and the eight new peaks from 12,210 to 13,507. The difference is several standard errors. The prediction that peak prevalence would move by less than 2% failed. Early growth matched. The height of the peak did not.

On the repaired file with the same cinema, church and hospital splits as W3, the shape-0.03 arm has longest chain 147, mean offspring 0.359 and k 0.037. Chains above 10 are 570 against 604 expected, and chains above 50 are 49 against 47. The longest chain is below 202 and below 250.

The three-week occupancy file on the repaired map has restaurant fill median 1.19, 95th percentile 1.50 and maximum 2.07. Eighty-three percent of restaurants peak above their design capacity. The phase-3 rule, at most 1% above capacity, failed. A target fill of 1.20 asks the median venue to peak above capacity, so that rule could not pass at the fill the feasibility test required.

The volume-model cells, on `agents4.json` and `locations3.json`, calibrated to early R 1.39:

| cell | early R | early k | replicate k |
|---|---:|---:|---|
| full volume, mid air exchange | 1.32 | 1.54 | 1.31 to 2.21 |
| full volume, low | 1.24 | 1.38 | 1.12 to 1.80 |
| full volume, high | 1.23 | 2.03 | 1.74 to 2.65 |
| fill pinned | 1.33 | 3.34 | 3.05 to 3.77 |

The fill-pinned and full-volume cells do not overlap. Using the design capacity in the volume denominator widens the offspring distribution relative to a model that ignores fill. That is the separation the peak-anchored capacity file did not show.

## W9. Household-share levers

Three mechanisms on the new map, calibrated to early R 1.3, with levers from day 21 and 12 replicates. Household share is measured on the uncontrolled replicates of the same arm. Shape 1 with venues off reached a share of 0.424. The X2 series had stopped at 0.389.

| arm | household share | venue closure | household isolation | ventilation | venue minus household |
|---|---:|---:|---:|---:|---:|
| venue profile, strength 3 | 0.243 | 72.0% | 39.5% | 67.5% | +33 points |
| venue profile, strength 1 | 0.365 | 67.1% | 46.6% | 60.7% | +21 points |
| shape 1, venues off | 0.424 | 41.8% | 53.1% | 39.5% | −11 points |

On X2 the same comparison was +31 points at share 0.25, +20 points at 0.37, and −11 points at 0.39. The closure benefit moved by 1.9 points at the low share (72.0 against 73.9) and by 1.7 points at the middle share (67.1 against 65.4). Both are inside 5 points. The sign is still positive at 0.365 and negative at 0.424. The crossing sits between those two shares. A shape-0.5 pilot landed at 0.386 and was not given the lever replicates, because the arm above 0.39 was the one that was kept. The window 0.37 to 0.39 was therefore not sampled by a lever arm.

Ventilation, which halves the hazard in restaurants, cinemas, long shops and churches, tracks venue closure: 68%, 61% and 40% at the three shares.
