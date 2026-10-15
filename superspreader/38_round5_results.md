# ACT-049 note 38: round 5

15 October 2026. Instructions are note 37. Reads with notes 31, 33, 34 and 36. Nothing in `locations1.json`, `agents2.json`, `locations3.json` or `agents4.json` was edited. No new PanSim runs. The closed-group event mechanism, R5-2 in note 36, was not started.

Trace day is `timestamp // 144`. Days 1 to 28 are trace days 1 through 28, so day 0 is outside that window. An index case is infected on days 1 to 10 and has a known infector. Its offspring are the infections it causes on or before day + 14. "Above 5" means strictly more than 5. "At least 20" means 20 or more. Negative-binomial shares use that arm's own mean and method-of-moments k, with the sample variance divided by n − 1. A 90% interval is a replicate bootstrap, 1,000 draws, the 5th and 95th percentiles. In `38_r53_iid_vs_observed.csv`, `iid_low` and `iid_high` are the central 95% Poisson interval around the count from 100,000 independent chains, scaled to the arm's number of chains. Size −1 in that file is every chain longer than 200, pooled. The simulation seed is 20261015. No simulated chain reached the size cap of 5,000.

## Deviations, and errors found in the notes

1. The factor-1,200 arm is the 8-week sample in `results/round2/r36/a_r0.003_f1200`, which is the sample that still has event logs. Its early R is 1.473 and its early k is 0.191. That is not the 4-week matched-sample k. The run is locations1.json with agents2.json, coefficient 0.00033188, event rate 0.003, factor 1,200, types 5 and 8.
2. Note 37's list of types whose largest capacity is at least 1,000 (6, 7 in part, 9 in part, 14, 15, 44) omits types 1, 8, 10 and 13. On `locations3.json` the types with maximum capacity at least 1,000 are 1, 6, 7, 8, 9, 10, 13, 14, 15 and 44. Type 6 is not uniformly that large: its minimum is 300 and its maximum is 1,000. "In part" is read here as locations with capacity at least 1,000. Types 6, 14, 15 and 44 are taken whole, because the note does not mark them "in part". On the ward file the split drops the maxima of types 12 and 14 below 1,000.
3. The 0.77% five-seed figure in note 22 and the later 0.947% are the same arm: old map, flat shape 0.03, 40 replicates, directory `results/x21/x3/shape0.03`. The chains, the mean, k and the longest chain match. The 0.77% calculation is not in the current scripts. The file-order figure below replaces it.
4. The early k in notes 32 and 35 is a method of moments on the pooled index cases, not a maximum-likelihood fit and not the mean of the per-replicate estimates. For E the pooled method of moments is 0.297, the mean of the 12 replicate estimates is 0.304, and the maximum-likelihood k at the same mean is 0.319. The histogram is the same one. The two published figures are the two estimators.
5. "About 178,750 agents are active" in note 06 is the population left after the default `--disableTourists 1` drops type 9. It is not a further activity filter. Both agent files contain 184,745 people.
6. A small set of infected agents is not reachable from an introduction, because the infector links form a cycle. In old-map replicate 09, agents 68314 and 116332 each name the other. Those agents are in the offspring histograms and outside the chain counts. The chain counts match notes 32 and 35, so the cycles were already outside those counts. The old map has 143 such agents, the unsplit new map 152, the hospital split 280, W3 54, W10 226 and W8 130.
7. P50 wide and P100 wide are the last two rows of the PS1 table in note 17, "P50, shape 0.05" and "P100, shape 0.05". The directories are `P50a005` and `P100a005`. Every PS1 run has 70 daily rows, so the final count is the day-70 count.
8. In the hospital-split chains, all 4,789 type-15 infections are on trace day 0. They are the introductions recorded at `work_commuter_box`, not later transmission. The whole-run commuter share is that day-0 mass. Days 1 to 28 have none.

## 1. Predictions

| task | prediction | result |
|---|---|---|
| R5-1 (i) | at capacity 100 and factor 1,091, the share with at least 50 offspring is above 0.1% and at least 100 times the negative binomial | failed. The share is 0.0388% (5 of 12,883; interval 0.016% to 0.061%). That is 131 times the negative binomial, so only the absolute bar fails |
| R5-1 (ii) | E stays within a factor of 3 of its negative binomial at 5 offspring and at 20 | met. Exactly 5 is 0.98 times the negative binomial. At least 20 is 1.08 times. Above 5 is 0.95 times |
| R5-1 (iii) | among index cases with any event offspring, the mean event offspring is between 15 and 60 | met. The four event arms are 25.6, 23.1, 31.6 and 15.7. The factor-600 interval is 14.4 to 16.9 |
| R5-3 (i) | independent draws match bins 1, 2, 3–4, 5–9, 10–19 and 20–49 for the five shape-0.03 arms | failed in two cells. Old map, 20–49, ratio 0.86 (195 against a band from 198). Hospital split, size 2, ratio 1.07 (1,060 against a band to 1,052). The other 28 cells are inside |
| R5-3 (ii) | W8 matches as well, unlike the caregiving arm | failed for bin 3–4 only, ratio 1.06 (1,570 against a band to 1,562). Chains above 10 are 392 against 414. The caregiving check is 220 against 291 |
| R5-4 (i) | homes, schools (3 and 33) and workplaces (4 and 13) are at least 60% of days 1–28 | failed. The locations3 baseline is 57.5% (57.1 to 57.9). The hospital-split chains are 48.9% (47.5 to 50.5) |
| R5-4 (ii) | the named large-capacity types are under 10% of infections | failed on the baseline, at 11.9% (11.7 to 12.1). Type 15 is 0. Large type-7 shops are 2.7 points of the 11.9 |
| R5-4 (iii) | no non-home location is more than 1% of baseline days 1–28 | met. The largest is hospital `64148_1500`, 203 infections, 0.42% (0.31 to 0.54) |
| R5-5 | adjusted P25k remains at least 50% below E | failed. The adjusted ratio is 0.618 (0.588 to 0.650). The raw ratio is 0.370. P25k stays below E |

Prediction (i) failed, and the consequence in note 37 applies to the bar that was registered. The events section should not claim that at least 0.1% of index cases have 50 or more offspring. The same arm still has a far tail the negative binomial does not: 5 index cases against 0.038 expected. Its share above 5 is 1.26%, which is below McKee's 3% and below its own negative binomial, 7.15%. The maximum is 91. Nobody reached 100. This is not the published 67% / 30% / 3% split, and it is not a negative binomial either. It is a thin shoulder above 5 and a handful of event-sized counts.

The two chain-size misses are a few counts outside a wide band. They are not the caregiving pattern. W3, W10 and the unsplit new map are inside every listed bin.

## 2. R5-0. Bookkeeping

`agents2.json` and `agents4.json` each contain 184,745 people, the same count as `agents1.json`. By type the counts are 1: 3,830, 2: 5,854, 3: 12,033, 4: 9,318, 5: 9,154, 6: 74,839, 7: 4,727, 8: 58,991, 9: 5,999. Type 9 is named tourist in `inputConfigFiles/agentTypes.json`. `--disableTourists` defaults to 1, and `initAgents` skips type 9. The study commands do not set the flag, so the loaded population is 178,746. Note 06's "about 178,750" is that number. On the locations3 run one agent starts in D1, so the living count on the first morning is 178,745.

The early k reported for E is the method of moments on the 12,854 pooled index-case offspring counts: mean 1.385, variance with divisor n − 1, k = 0.2970. The mean of the twelve replicate-level estimates is 0.3041. A maximum-likelihood fit with the mean held at 1.385 gives k = 0.3186. That is the 0.319 figure. The histogram in `35_r45_offspring_hist.csv` was rebuilt from the traces and matches. Notes 32 and 35 should keep the label "method of moments" on 0.297.

The five-seed probability is the chance that five introductions have total size above 50. Note 22's 0.77%, against a negative binomial of 0.95%, is the old-map shape-0.03 arm: 38,545 chains, mean 0.363, k 0.034, longest 254. Grouping those stored chains in file order, non-overlapping groups of five, discarding the leftover chains, gives 7,709 groups and 0.946945%. The Dwass probability at mean 0.363450 and k 0.033518 is 0.943811%. Recomputing the grouping from the traces reproduces 0.946945%. Drop 0.77%. The arm is shape 0.03 on the old map.

Flags added or changed since round 3:

| flag | default | commit | what the code does |
|---|---|---|---|
| `--eventMaxCapacity` | 0 | `c51b3ad`, 15 October 2026 | If positive, a location can host an event only when its type is allowed and its capacity is at or below this value. Zero leaves eligibility and the random draws unchanged |
| `--homeVenueFactor` | −1 | flag added in `0184e68`, 14 October 2026 | If non-negative, this replaces the venue-profile base of type 2 before `venueStrength` is applied |
| guard on that factor | | `c51b3ad` | The factor is applied when the profile is on, or `communityScale` is not 1, or the factor is non-negative. Before this commit an explicit home factor was ignored while the profile was off. The default, −1 with the profile off, is unchanged |

The three files are `include/policies/infectionPolicies.h`, `include/locationList.h` and `src/runtime/locationListRuntime.cpp`.

## 3. R5-1. Event offspring against McKee and against the negative binomial

McKee 2024, among 9,591 index cases: 67% infected nobody, 30% infected 1 to 5, 3% infected more than 5, and the largest was 101. The arms below were already run. The baseline is the R3-6 no-event arm, day-28 mean 6,211.25, 4 weeks, locations1.json and agents2.json, coefficient 0.00041. The negative binomial is evaluated at the mean and k in the same row. `38_r51_event_offspring.csv` is the joint histogram of total offspring and offspring infected at an event location on an event day. `38_r51_summary.csv` has the intervals.

| arm | index cases | mean | k | 0 | 1 to 5 | above 5 | at least 20 | at least 50 | at least 100 | maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-event baseline | 13,414 | 1.408 | 4.59 | 0.297 | 0.691 | 0.0122 | 0 | 0 | 0 | 11 |
| negative binomial | | | | 0.293 | 0.696 | 0.0110 | | | | |
| uncapped, factor 1,200 | 12,424 | 1.473 | 0.191 | 0.346 | 0.640 | 0.0146 | 0.00660 | 0.00145 | 0 | 93 |
| negative binomial | | | | 0.662 | 0.255 | 0.0829 | 0.00746 | 0.000104 | | |
| capacity 100, factor 1,091 | 12,883 | 1.388 | 0.310 | 0.339 | 0.649 | 0.0126 | 0.00388 | 0.000388 | 0 | 91 |
| negative binomial | | | | 0.590 | 0.338 | 0.0715 | 0.00221 | 0.00000296 | | |
| capacity 100, factor 2,500 | 13,900 | 1.448 | 0.128 | 0.363 | 0.624 | 0.0136 | 0.00705 | 0.00223 | 0.000216 | 141 |
| negative binomial | | | | 0.726 | 0.194 | 0.0802 | 0.01207 | 0.000503 | | |
| capacity 100, factor 600 | 12,440 | 1.399 | 0.641 | 0.326 | 0.660 | 0.0143 | 0.00257 | 0 | 0 | 47 |
| negative binomial | | | | 0.476 | 0.471 | 0.0530 | 0.000187 | | | |
| E, shape 0.3 | 12,854 | 1.385 | 0.297 | 0.585 | 0.346 | 0.0685 | 0.00265 | 0.000078 | 0 | 63 |
| negative binomial | | | | 0.597 | 0.330 | 0.0722 | 0.00246 | 0.00000408 | | |
| McKee 2024 | 9,591 | | | 0.67 | 0.30 | 0.03 | | | | 101 |

At factor 1,091 the negative binomial puts almost none of the mass at 50 offspring (about 0.038 of 12,883 index cases). The traces have 5. That ratio, 131, clears the factor of 100. The share itself, 0.0388%, does not clear 0.1%. Both halves were required, so prediction (i) fails. Factor 2,500 is the arm that passes the absolute bar: 0.223% have at least 50 offspring (interval 0.142% to 0.295%) and the maximum is 141, but that arm's k is 0.128, not 0.3. Factor 600, k 0.641, has no index case at 50 or more and a maximum of 47.

E is the row that sits on its negative binomial through the body. Exactly 5 offspring is 0.0232 against 0.0236. At least 20 is 0.00265 against 0.00246. One index case has 63 offspring, so the at-least-50 ratio is 19 and the interval on that share includes 0. Prediction (ii) did not cover that cell.

Event offspring, among index cases who have at least one:

| arm | index cases with any | mean event offspring | 90% interval | share of all their offspring that were at events |
|---|---:|---:|---|---:|
| uncapped, factor 1,200 | 144 | 25.6 | 21.9 to 29.4 | 0.201 |
| capacity 100, factor 1,091 | 98 | 23.1 | 21.8 to 24.5 | 0.127 |
| capacity 100, factor 2,500 | 146 | 31.6 | 28.6 to 34.1 | 0.229 |
| capacity 100, factor 600 | 106 | 15.7 | 14.4 to 16.9 | 0.095 |

All four means are inside 15 to 60. Prediction (iii) is met. The baseline and E have no events, so the event count is zero.

## 4. R5-3. Chain-size body

The six arms are the histogram in `35_chain_size_histograms.csv`. Offspring here is the full trace, every infected agent, not the early window. `38_r53_offspring_hist_all.csv` is that histogram. Independent chains draw from it with replacement. The caregiving row is the check against note 29, not one of the six arms: chains above 10 were 220 observed and 293 under independent draws. This draw gives 220 and 291.

| arm | chains | mean offspring | k | longest | Spearman | 90% interval | parents |
|---|---:|---:|---:|---:|---:|---|---:|
| old map, shape 0.03 | 38,545 | 0.363 | 0.0335 | 254 | 0.387 | 0.365 to 0.409 | 4,907 |
| new map, hospitals unsplit | 38,613 | 0.397 | 0.0303 | 585 | 0.385 | 0.365 to 0.406 | 5,381 |
| hospitals split | 38,499 | 0.345 | 0.0315 | 150 | 0.381 | 0.358 to 0.402 | 4,716 |
| W3 | 38,583 | 0.311 | 0.0351 | 202 | 0.377 | 0.352 to 0.400 | 4,460 |
| W10 | 38,879 | 0.359 | 0.0374 | 147 | 0.383 | 0.366 to 0.400 | 4,994 |
| W8, home weight 0.35 | 38,765 | 0.328 | 0.0966 | 61 | 0.289 | 0.270 to 0.309 | 7,660 |
| V5 caregiving, check only | 38,531 | 0.319 | 0.145 | 44 | 0.208 | 0.188 to 0.228 | 8,386 |

Spearman correlates a parent's offspring count with the mean offspring count of its children, average ranks for ties, and only parents with at least one child. It excludes the cycle components in deviation 6. The correlation is positive in every arm, and the flat arms are higher than the caregiving arm. Generation means are in `38_r53_correlation.csv`. They stay near the arm's own mean through the generations that have more than a few dozen nodes. They do not fall steadily.

Bins, observed count against the independent-draw expectation. A cell is outside when the observed count is outside the Poisson band.

| arm | 1 | 2 | 3–4 | 5–9 | 10–19 | 20–49 |
|---|---:|---:|---:|---:|---:|---:|
| old map | 0.999 | 1.014 | 1.029 | 1.033 | 0.975 | 0.859 outside |
| new map, unsplit | 1.001 | 0.977 | 1.042 | 0.969 | 0.946 | 1.006 |
| hospitals split | 0.998 | 1.071 outside | 1.000 | 1.033 | 0.996 | 0.902 |
| W3 | 0.998 | 1.059 | 1.007 | 0.996 | 1.001 | 1.005 |
| W10 | 0.999 | 1.029 | 1.020 | 1.026 | 0.954 | 0.998 |
| W8 | 0.997 | 1.000 | 1.056 outside | 1.019 | 0.975 | 1.020 |
| caregiving, check | 0.988 outside | 1.040 | 1.139 outside | 1.090 outside | 0.803 outside | 0.744 |

The old-map 20–49 cell is 195 chains against 227 expected, band 198 to 257. The hospital-split size-2 cell is 1,060 against 990, band 929 to 1,052. W8's 3–4 cell is 1,570 against 1,486, band 1,411 to 1,562. Prediction (i) fails in the first two of those, and prediction (ii) fails in the third. The caregiving arm fails four of the six bins, and chains above 10 are 220 against 291. W8's chains above 10 are 392 against 414, inside the two bins that cover 10 to 49. The body departure from the negative binomial, in the five flat arms and in W8, is carried by the offspring distribution itself, except for those three cells.

## 5. R5-4. Where the early infections are

The open-epidemic arm is R3-2 (c): `locations3.json`, `agents4.json`, coefficient 0.00041, 16 weeks, 8 replicates. The chain arm is R3-4: the ward location file, shape 0.03, 7 weeks, 40 replicates. Trace location indexes follow the place order in the file that was simulated. Two location ids are repeated in each file (`work_commuter_box` is stored three times); the index in the trace is the slot that was infected. No infection had a location index past the end of the file, and no trace type disagreed with the file type at that index.

Days 1 to 28 on the locations3 baseline, 48,478 infections:

| group | share | 90% interval |
|---|---:|---|
| home (2) | 0.448 | |
| school (3) plus classroom (33) | 0.116 | |
| work (4) plus night work (13) | 0.011 | |
| those together | 0.575 | 0.571 to 0.579 |
| university (44), not in the prediction | 0.022 | |
| note's large-capacity list | 0.119 | 0.117 to 0.121 |
| any location with capacity at least 1,000 | 0.112 | 0.109 to 0.115 |
| every type whose maximum is at least 1,000 | 0.312 | 0.307 to 0.316 |

Homes, the named schools and the named workplaces miss 60%. Adding the university still leaves the point estimate at 59.7%. The next shares are long shops 9.2%, short shops 7.8%, hospitals 5.1%, cinemas 4.4% and restaurants 4.2%.

The note's list sums cinema 4.35%, health 2.62%, university 2.20%, type-7 locations of capacity at least 1,000 at 2.66%, and type-9 locations of that size at 0.05%. Type 15 is zero. Without the large short shops the list would be 9.2%, under 10%. The crossing is not the commuter box. It is the list as a whole, and the large short shops are part of it. Long shops, omitted from the list, are 9.2% by themselves, and parks are 3.7%. That is why the complete set of types with a large maximum is 31%, not 12%.

The largest non-home location is `64148_1500`, type 12, capacity 750, 203 infections, 0.42% of the window. The next locations are other hospitals and two nursing homes, each under 0.42%. Prediction (iii) is met. Over the whole run the largest non-home share is a cinema, `7098_2322`, at 0.26%.

On the hospital-split chains, days 1 to 28, homes plus the named schools and workplaces are 48.9%. Short shops are 12.5% and long shops 11.5%. The note's large-capacity list is 15.0%, and locations of capacity at least 1,000 are 13.8%. Type 15 is zero in the window. Over the whole run the commuter share is 8.1%, and it is entirely the day-0 introductions at `work_commuter_box` (deviation 8).

`38_r54_infections_by_type.csv` has every type, both windows, both arms. `38_r54_top_locations.csv` has the 20 locations with the most infections in each window.

## 6. R5-5. PS1 at day 28 and day 70

Daily tables were kept, so nothing was re-run. The files are `results/ps/ps1/<arm>/rep1101` through `rep1112`, on the old map (`agents1.json`, `locations0.json`), 10 weeks. `38_r55_ps1_cumulative.csv` has one row per replicate. The regression is within-arm: final count on the day-28 count, with a separate intercept for each arm. The slope is 4.00 extra final infections per extra day-28 infection (interval 3.57 to 4.42). The within-arm R² is 0.72. Adjusted values are shifted to E's mean day-28 count, 5,060. Note 33's slope of 3.5 was a different set of runs, and the day-112 slope in note 35 was 1.23.

| arm | day 28 | day 70, which is also the final | adjusted to E's day 28 | adjusted / E |
|---|---:|---:|---:|---:|
| A0 | 5,918 | 59,574 | 56,146 | 1.163 |
| E | 5,060 | 48,274 | 48,274 | 1.000 |
| P0 | 5,532 | 50,684 | 48,796 | 1.011 |
| P25 | 4,014 | 38,695 | 42,876 | 0.888 |
| P50 | 2,567 | 21,808 | 31,776 | 0.658 |
| P100 | 1,862 | 12,680 | 25,466 | 0.528 |
| P25k | 2,070 | 17,855 | 29,811 | 0.618 |
| P100ln | 3,975 | 26,050 | 30,387 | 0.629 |
| P50 wide | 1,035 | 6,984 | 23,076 | 0.478 |
| P100 wide | 820 | 4,220 | 21,169 | 0.439 |

P25k's raw final size is 37.0% of E's. Its day-28 count is 2,070 against E's 5,060, even though its early R in note 17 (1.377) is higher than E's (1.333). Early R is offspring per early infector. The day-28 count is the size of the epidemic, and P25k's is smaller. Moving P25k up to E's day-28 count, at 4.00 infections each, closes part of the gap and leaves 61.8% (interval 58.8% to 65.0%). Prediction R5-5 fails. It is not a reversal: adjusted P25k remains below adjusted E, by 38% rather than by 50% or more. P0 stays next to E after the same adjustment, ratio 1.011.

## Files

- `38_r51_event_offspring.csv`, `38_r51_summary.csv`
- `38_r53_offspring_hist_all.csv`, `38_r53_iid_vs_observed.csv`, `38_r53_correlation.csv`
- `38_r54_infections_by_type.csv`, `38_r54_top_locations.csv`
- `38_r55_ps1_cumulative.csv`
- `study/analyze_round5.py` reads the stored traces and the PS1 daily tables. It does not run PanSim.
