# Note 25 experiments on locations1.json and agents2.json

13 October 2026. Protocol: `25_venue_subdivision_review_and_experiments.md`. The superseded v1 files were not used. V6, V7 and V8 were not run. V6 is the event sweep, V8 is the lever repeat, and V7 stays conditional: the new venues have design capacities, but area is still zero for types 5, 9 and 10, so an area-based volume model is still not fed by the file.

## V0. What the new files actually are

Agent count is unchanged at 184,745. The number of schedule slots of every type is unchanged, including three type-5 slots for 153,710 agents, three type-9 slots for 163,028, and three type-10 slots for 166,858. Time use did not change. Only the location ids on types 5, 9 and 10 were reassigned, each agent to the nearest distinct places. Assignment is a fixed subset, not “every agent is a candidate for every venue.”

| type | locations old → new | median agents assigned, old → new | median peak occupancy, old → new | area |
|---|---:|---:|---:|---|
| 5 restaurant and pub | 8 → 420 | 57,650 → 944 | 3,897 → 69 | zero, old and new |
| 9 church | 22 → 52 | 22,263 → 7,783 | 1,430 → 534 | zero |
| 10 park | 15 → 167 | 33,375 → 2,735 | 1,288 → 120 | zero |
| 6 cinema | 20 → 20 | | 1,406 → 1,402 | zero |

Restaurant median peak 69 is the note’s predicted 73. The busiest restaurant still reaches 310, because assignment is uneven (5 to 4,022 agents). Church median peak falls to 534, still above the type’s 50–500 label, and the busiest church still reaches 1,588. Cinemas are untouched. New capacities are design values (cafes 40, restaurants 100, added churches 150, added parks 300). The infectiousness field is 1 for every new place, the same as the old places. Seven of 420 restaurants and one park have no assigned agent.

## V1. The aggregate fit does not move

Both maps accept the same infection coefficient, 0.00041, at a two-seed early R of 1.377. That is a change of 0%, inside the pre-registered 5%. Eight replicates:

| | old map | new map |
|---|---:|---:|
| early R | 1.413 | 1.406 |
| peak prevalence | 13,662 | 13,526 |
| share of infections in restaurants | 4.43% | 4.28% |
| churches | 1.42% | 1.41% |
| parks | 3.76% | 3.66% |
| cinemas | 4.58% | 4.57% |
| homes | 46.26% | 46.33% |

No type moves by as much as 2 percentage points. Peak prevalence moves by 1%. Four 37-week runs on the new map, with the existing variant-infectiousness vector left as it was, give a 105-day hospital area of 10,916 (range 9,893 to 11,597), against the old eight-replicate baseline of 10,187. The vector was not re-fitted.

## V2. Early k is unchanged. Final size is not, because growth drifted

Twelve replicates, venue mechanisms off, each arm calibrated to early R 1.39 on two seeds:

| arm | twelve-replicate early R | pooled early k | final size | old early k | old final size |
|---|---:|---:|---:|---:|---:|
| A0 | 1.382 | 5.23 | 58,613 | 4.66 | 59,574 |
| E | 1.286 | 0.313 | 36,229 | 0.311 | 48,274 |
| P0 | 1.408 | 0.315 | 68,579 | 0.295 | 50,684 |

A0’s k, 5.23, is 12% above 4.66, inside the 15% band. E and P0 remain indistinguishable on k (0.313 against 0.315). Their final sizes are not within 10% of the old figures, and they are not close to each other, because the twelve-replicate early R drifted apart (1.29 against 1.41) after two-seed calibrations of 1.43 and 1.36. The venue file does not show up in the dispersion comparison. It does show up as a less stable match of growth between the per-episode and persistent arms, and final size follows that.

## V3. The ceiling test still fails at shape 0.3, and it meets the thresholds at shape 0.03

The distinct-contact field from section 6 was not added. Location id was already in the trace. The pre-registered offspring test was run at 8 replicates.

At shape 0.3 the original lower-half rule gives a top-1% ratio of 0.62 (bootstrap 0.52 to 0.72) and an elasticity of 0.71. On the old map the same rule gave 0.49. The ratio did not fall, and the test does not pass. Decile means still rise (top decile: multiplier 5.4, offspring 6.2; top 1%: multiplier 11.6, offspring 14.3).

At shapes 0.08 and 0.03 the median multiplier is essentially zero, so the lower-half slope is not a usable line. The slope from the moderate bin, multipliers 0.5 to 1, is the same bin the elasticity uses:

| shape | top-1% offspring | proportional prediction | ratio | bootstrap | elasticity |
|---|---:|---:|---:|---|---:|
| 0.08 | 31.6 | 49.7 | 0.64 | 0.57 to 0.71 | 0.63 |
| 0.03 | 55.2 | 120 | 0.46 | 0.40 to 0.51 | 0.49 |

Shape 0.08 stays above the 0.60 thresholds. Shape 0.03 is below all three (ratio, upper bound, elasticity). That is the outcome the note said would mean the shortfall acts through the single-venue-day extreme rather than through the episode budget. It is not the original lower-half test, because that test has no slope when half the multipliers are zero. The top 1% still produces 55 offspring. The shortfall is against a straight line, not a hard stop at zero.

## V4. The Nipah-like chain match survives the new restaurants

Shape 0.03, flat multiplier, 40 replicates on the new map, coefficient calibrated to mean offspring 0.33. The full sample drifted to a higher mean, as it did on the old map.

| | new map | negative binomial at that mean and k | old aggregated map |
|---|---:|---:|---:|
| chains | 38,613 | | 38,545 |
| mean offspring | 0.397 | | 0.363 |
| k | 0.030 | | 0.034 |
| zero share | 0.916 | | 0.919 |
| chains > 10 | 642 | 665 | 540 |
| chains > 50 | 71 | 82 | 61 |
| maximum | 585 | | 254 |
| five-seed outbreaks above 50 | 1.2% | | 0.77% |

Chains above 50 do not collapse. They sit with the negative binomial. k stays near 0.03. A negative binomial at the literature pair (0.33, 0.036) puts 0.61% of five-seed outbreaks above 50 cases; this arm puts 1.2% there, at a higher realised mean. Section 5.10 holds on the subdivided restaurant map. Cinemas are still one room of about 1,400 people, and the longest chain is 585, so the untouched type-6 aggregate is still in the tail.

## V5. Splitting restaurants does not remove the remaining thin middle

The two subcritical arms, hospitals and health centres also split into wards of about 30, on the new map. About 38,500 chains each.

| arm | mean | k | chains > 10 | NB expected > 10 | chains > 50 | max |
|---|---:|---:|---:|---:|---:|---:|
| flat, shape 0.04 | 0.321 | 0.049 | 485 | 499 | 19 | 99 |
| caregiving, shape 0.09 | 0.319 | 0.145 | 220 | 313 | 0 | 44 |
| caregiving, X1, restaurants still aggregated | 0.317 | 0.151 | 195 | 301 | 0 | 42 |

The flat arm follows its negative binomial. The caregiving deficit of mid-sized chains is 220 against 313, a gap of 93, against 106 on the X1 map where restaurants were still aggregated. That is not a halving. Restaurant-to-restaurant inheritance was already 1.8% in the old caregiving arm (204 restaurant-infected cases) and is 0% here. The thin middle that survived the hospital split also survives the restaurant split. It is in the schedule, not in the aggregated restaurants.

## What this does to the conclusions

The person-level dispersion results and the aggregate fit are unaffected. The contact-ceiling rejection stands at shape 0.3 and is no longer uniform at shape 0.03, where a moderate-slope test now meets the old thresholds. The Nipah-like chain match in section 5.10 survives the new restaurants. The leftover thin middle of the caregiving chains is not a restaurant-aggregation artefact. Churches are improved and still too big, and cinemas are unchanged.
