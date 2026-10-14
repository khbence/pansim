# ACT-049 note 32: round 3

14 October 2026. Instructions are note 31, which replaces section 7 of note 30. No edits were made to `locations1.json`, `agents2.json`, `locations3.json` or `agents4.json`. Day 28 is the 28th row of the daily table. "Above" means size strictly greater than the threshold. Negative-binomial expectations are the Dwass total-progeny probabilities at each arm's own mean and k. Intervals are replicate bootstraps.

## 1. Predictions

| task | prediction | result |
|---|---|---|
| R3-2 | one of three accounts of the 5.4% peak fall | the third: neither the permutation nor the repair falls, once day-28 incidence is matched and the run is long enough for the peak to pass. The 5.4% is withdrawn. The permutation rises 5.4% |
| R3-3 volume | fill-pinned and full-volume mid still separate by more than their replicate ranges | met. Replicate k is 2.34–3.60 against 0.99–1.75 |
| R3-3 waves | refit the variant vector only if the hospital difference exceeds the Welch 80% minimum detectable difference | not exceeded. Area difference −1,117 against a minimum of 1,767. Peak difference +12.5 against a minimum of 44. Vector unchanged |
| R3-4 | longest chain at most 300 and chains above 200 inside the Poisson band: the 585 was a hospital artefact alone | met. Longest chain 150. Chains above 200: 0, expected 0.23 |
| R3-5 | early k within 13%; final size equivalent if the 90% interval of the ratio is inside 0.90–1.10 | k ratio of P0 to E is 1.003. Final-size ratio 1.035, 90% interval 0.991–1.084, inside the band. The pre-registered "entirely outside" finding did not occur |
| R3-6 | early k of 0.3 between factors 1,100 and 1,300 in configuration (a); median non-empty cluster below 10 | the factor is met (k 0.292 at 1,200x). The cluster median is 28, so that part failed |
| R3-7 | venue closure minus household isolation changes sign between household shares 0.37 and 0.39 | failed. The shape-0.5 arm at share 0.386 is already negative (−11 points), and so is the arm at 0.367 |

## 2. R3-1. Tables from the existing runs

Chain counts for W3, W8, W10 and the old-map shape-0.03 arm are in `32_chain_counts.csv`, with negative-binomial and empirical-iid rows and the five-seed probability of a total above 50. W3 reproduces the earlier figures: 454/25/2/1/0 chains above 10/50/100/200/500, longest 202, five-seed 0.428% against an exact 0.484%. W10 is 570/49/3/0/0, longest 147, five-seed 0.836% against 0.811%. The old map (X3, the old-map shape-0.03 arm) has longest chain 254, 14 chains above 100 and 1 above 200.

`32_long_chain_composition.csv` has the longest chain of each of those arms, and `32_location_names.csv` names every `largest_location_index` in that file and in `29_long_chain_composition.csv`. The 585 chain's largest location, index 75375, is `64118_1496`. The W3 chain of 202 has its largest location at `121284_1803` (type 8, capacity 3,000, 24.8% of the chain); the chain is 44% shop and 0% hospital. The W10 chain of 147 has its largest location at `6a_42937altal` (a classroom, 10% of the chain); the chain is 47% school and 33% home, and 0% hospital.

`32_w10_presence.csv` is the 420 type-5 venues. 397 are stable and 23 use the median presence rate 0.0769. `32_w10_feasibility.csv` gives the sums. At fill 1.10 the continuous sum is 441,144 and the floored quota is 440,967, short of 461,130. At 1.15 the continuous sum is 461,196 (66 above the assignments) and the floored quota is 460,982, short by 148, which is not 2% slack. At 1.20 the floored quota is 481,077, slack 19,947, 4.3%. The occupancy audit on the repaired file, with the note 28 thresholds scaled by 1.20/1.10, passes the median (1.19, band 0.55–1.20) and fails the 95th percentile (1.50 against 1.42) and the maximum (2.07 against 1.75).

`32_peak_prevalence.csv` is the eight-week comparison that note 29 reported. Every replicate peaks on the last day, so those peaks are prevalence at day 56 while the epidemic is still climbing. `32_w6c_hospital.csv` is the eight new-map wave replicates. `32_w9_levers.csv` is the three existing lever arms; the reductions match note 29 (72.0%, 39.5%, 67.5% at share 0.243; 67.1%, 46.6%, 60.7% at 0.365; 41.8%, 53.1%, 39.5% at 0.424). `32_w4_top1.csv` and `32_offspring_curve.csv` are the re-binned curves. The missing (0.3%, 2000x) row is now the last line of `29_events.csv`: early R 1.380, k 0.121, 3,425 events, 90th percentile 53, maximum 1,796, type 8, event share 0.184.

### Reassignment

`32_reassignment_summary.csv`. Displacement in the solver, and in note 29, is the per-agent mean of the three new distances minus the mean of the three old distances. Distance is the minimum of the distance to home and the distance to work. Sorting the three slots before subtracting does not change this mean. On all 153,710 agents that 95th percentile is 2,002 m and the median is 33 m, which is the reported figure. The share whose venue set changed is 57.5%.

The first 6,416 agents, which is the slice note 31 could read, reproduce note 31 exactly: agent-mean 95th percentile 3,835 m, slot-level 95th percentile 3,966 m, 4.7% of slots above 5 km, old median distance 306 m, new median 584 m, old 95th percentile 1.56 km, new 4.94 km, 60.0% changed. Among changed agents in that slice the 95th percentile is 4.46 km. The two definitions agree. The id-ordered prefix is not representative of the file. On the full file the new assigned distance has median 286 m against an old median of 156 m, and a 95th percentile of 2.43 km against 0.84 km. The share of slots above 5 km is 0.44%, and the maximum agent-mean displacement is 5.29 km. The 5 km escalation rule uses the 95th percentile, which is 2.0 km.

### Offspring curve

At shape 0.3 on the new map, offspring per unit multiplier across deciles of positive multipliers runs 1.81, 1.89, 1.72, 1.68, 1.60, 1.39, 1.14 from the fourth decile to the tenth. The top 5% is 1.10 and the top 1% is 1.22 (interval 1.06–1.38). The reference-free ratio at the top 1% is 0.907 (0.819–0.997) on the new map at shape 0.3, 0.869 (0.761–0.972) at 0.08, and 0.857 (0.787–0.939) at 0.03. The old map is 0.784, 0.907 and 0.768 at shapes 0.08, 0.03 and 0.3. The ceiling rule needs the ratio below 0.75 and the upper bound below 0.85. No shape meets it, including the top 1% bins that the log grid could not resolve. The first decile is not usable: its mean multiplier is near zero and the interval runs from 0 to 42.

## 3. R3-2. The peak, matched on day-28 infections

Eight replicates, open epidemic, 16 weeks, baseline settings. The peak is on day 66–73 and prevalence at day 112 is about a quarter of the peak, so the peak is no longer the last day of the run. Arm (a) is `locations1.json` with `agents2.json` at coefficient 0.00041. Arm (b) is the same locations with type-5 ids shuffled across slots, seed 20261014, each venue keeping its assigned count; 1,795 swaps removed within-agent duplicates and none remained. Arm (c) is `locations3.json` with `agents4.json`. Day-28 infections are 5,808, 5,806 and 5,814. Arm (b)'s accepted coefficient is 0.00041505, from the third round. Arm (c) matched at 0.00041 on the first round. Per-replicate values are in `32_r32_peak.csv`.

| arm | peak prevalence | peak day | final size | 105-day hospital area |
|---|---:|---:|---:|---:|
| (a) current file | 16,375 (SD 217) | 66–73 | 97,687 | 41,147 |
| (b) permuted restaurants | 17,256 (SD 275) | 66–73 | 100,487 | 42,761 |
| (c) repaired file | 16,495 (SD 387) | 66–73 | 98,016 | 41,540 |

Arm (b) is 5.4% above (a), standard error 124. Arm (c) is 0.7% above (a), standard error 157. Neither falls. The third of the three registered accounts is the one that holds, and the 5.4% fall from the eight-week runs is withdrawn: those runs ended on day 56, while prevalence was still at its maximum for the run. The permutation does not match (a). It is higher, by several standard errors, so the registered overlap account (a fall of at least 3% from breaking co-attendance) does not hold either. The repair matches the current file on the peak, the final size and the hospital area.

## 4. R3-3. Waves, then volume

Eight 37-week replicates on the repaired file, coefficient 0.00041, variant vector unchanged (`32_r33_waves.csv`). Wave-1 hospital area has mean 9,070 (SD 1,125) against 10,187 (SD 1,218) on the eight-replicate old-file baseline. The difference is −1,117 and the Welch 80% minimum detectable difference is 1,767. The wave-2 peak after day 140 has mean 240 (SD 20) against 227 (SD 35). The difference is +12.5 and the minimum detectable difference is 44. The vector was not refit. The same comparison against the eight new-map W6(c) replicates gives area −1,129 against a minimum of 1,703, and peak +16 against a minimum of 39.

The four volume cells were then matched on day-28 infections to the full-volume mid cell (mean 5,461). The other three landed at 5,386, 5,445 and 5,393, a spread of 1.4%. Twelve replicates, `32_r33_volume.csv`.

| cell | coefficient | day 28 | early k | replicate k |
|---|---:|---:|---:|---|
| full volume, mid | 0.00034273 | 5,461 | 1.34 | 0.99–1.75 |
| full volume, low | 0.00022566 | 5,386 | 1.19 | 0.91–1.70 |
| full volume, high | 0.00051786 | 5,445 | 1.55 | 1.27–2.08 |
| fill pinned | 0.00031546 | 5,393 | 2.86 | 2.34–3.60 |

Fill-pinned and full-volume mid do not overlap. The prediction is met. The earlier separation was not a growth artefact. Low and high air exchange do overlap the mid cell.

## 5. R3-4. Hospital split alone

The shape-0.03 arm on the new map, with only hospitals, health centres and nursing homes split into wards of about 30. Cinemas and churches stay aggregated. Forty replicates. The four-seed calibration measured mean offspring 0.329 at coefficient 0.00015393. An eight-replicate probe at that coefficient was 0.368, so the reported run uses 0.00013799. The forty-replicate mean is 0.345 and k is 0.0315, rather than the target 0.33. Expectations below use 0.345.

Chains above 10, 50, 100, 200 and 500 are 523, 49, 7, 0 and 0, against negative-binomial expectations 562, 48, 6.4, 0.23 and 0 (`32_r34_chain_counts.csv`). Chains above 200: 0, expected 0.227, and the Poisson band of an observed 0 runs up to 3.69. The longest chain is 150. Both halves of the prediction hold, so the 585 chain was a hospital artefact alone. The longest chain is 35% shop, 21% school, 18% home, 10% cinema and 2% church, and the hospital share is 0. Its largest location is `40588_1996`, a type-8 shop of capacity 50, holding 23%. The failure branch (longest chain above 400, or two chains above 200 against an expectation below 0.2) did not occur.

## 6. R3-5. Final size at matched day-28 incidence

Twelve replicates, 10 weeks, on the new map. A0 at coefficient 0.00041 has mean day-28 infections 6,221. E matched on the first round (6,124, coefficient 0.00049784). P0 matched on the third (6,299, coefficient 0.00049726). Per-replicate values are in `32_r35_final_size.csv`.

| arm | day 28 | early R | early k | zero share | top 20% share | final size |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 6,221 | 1.384 | 4.82 | 0.300 | 0.504 | 60,337 (SD 2,258) |
| E | 6,124 | 1.385 | 0.297 | 0.585 | 0.801 | 56,629 (SD 5,000) |
| P0 | 6,299 | 1.400 | 0.298 | 0.580 | 0.796 | 58,587 (SD 2,481) |

The k ratio of P0 to E is 1.003, inside 13%. The final-size ratio of P0 to E is 1.035. Its 90% replicate-bootstrap interval is 0.991–1.084, which is inside 0.90–1.10, so equivalence holds. The 14.5% gap in note 29 does not survive matching on day-28 infections. The pre-registered finding, that the 90% interval would lie entirely outside 0.90–1.10, did not occur. E against A0 is 0.939, with a 90% interval 0.898–0.979: the interval crosses 0.90, so it is neither inside the equivalence band nor entirely outside it.

## 7. R3-6. Events

The baseline is the no-event open epidemic on the current file, twelve replicates, mean day-28 infections 6,211. Configuration (a) at 1,200x, 1,500x and 2,000x matched that mean (6,138, 6,179 and 6,249). Early k below is from those matched replicates. Non-empty cluster percentiles are from a separate eight-week sample at the same coefficient; that sample's own day-28 mean is also listed, because a fresh twelve replicates move by a few percent.

| arm | factor | matched day 28 | early k on the matched sample | eight-week day 28 | non-empty median | non-empty 90th | maximum | max type |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) | 1,200 | 6,138 | 0.292 | 6,261 | 28 | 104 | 2,086 | 8 |
| (a) | 1,500 | 6,179 | 0.230 | 6,376 | 32 | 115 | 2,339 | 8 |
| (a) | 2,000 | 6,249 | 0.171 | 6,539 | 32 | 128 | 2,473 | 8 |
| (b) | 100 | 6,274 | 2.61 | 5,907 | 7 | 27 | 414 | 6 |
| (b) | 1,000 | 6,184 | 0.198 | 5,395 | 23 | 107 | 1,501 | 6 |

Early k of 0.3 sits at a factor of 1,200 (k 0.292), inside 1,100–1,300. The median non-empty cluster there is 28, not below 10. That part of the prediction failed. The eight-week sample at 2,000x has day-28 infections 5.3% above the baseline, so its early k (0.115) is not the matched-sample figure.

Configuration (b) missed the band in three secant rounds and was finished by interpolating in log k between the bracketing rounds. At 100x the matched sample has day-28 infections 6,274 and early k 2.61. At 1,000x it has 6,184 and early k 0.198. The eight-week samples at those coefficients have day-28 means 5,907 and 5,395. The 1,000x calibration sample itself runs from 4,196 to 10,822, so a second draw of twelve replicates is not pinned to 3%. Early k on the eight-week 1,000x sample is 0.176, close to the matched-sample 0.198. Non-empty medians are 7 at 100x and 23 at 1,000x. The maximum is a cinema in both (414 and 1,501). All five arms are in `32_r36_events.csv`.

## 8. R3-7. The household-share window

The shape-0.5 arm, coefficient 0.00046062 from the W9 calibration, now has twelve lever replicates. Household share on the uncontrolled replicates is 0.386. Venue closure reduces final size by 39.4%, household isolation by 50.3%, ventilation by 37.6%. Venue minus household is −10.8 points (interval −12.4 to −9.3).

An arm with shedding shape 0.35 and venues off, coefficient 0.00042728, was aimed at a share of 0.375. The two-seed pilot was 0.371. The twelve-replicate share is 0.367. Venue closure reduces final size by 48.4%, household isolation by 59.0%, ventilation by 43.6%. Venue minus household is −10.6 points (interval −12.8 to −8.3).

The prediction required the sign change between 0.37 and 0.39, and the conditional was that the arm at 0.386 would be positive. It is negative, and the arm at 0.367 is negative as well. The venue-profile arm at share 0.365, from W9, remains positive at +20 points. The sign is not a function of household share alone across these mechanisms. The registered window was not where the sign changed.

## 9. What this changes for the wording

Nothing in the paper was edited. From these runs: section 5.5 can say that offspring per unit multiplier falls gradually across the deciles at shape 0.3, from about 1.8 to 1.14, and that the top 1% at shapes 0.08 and 0.03 has reference-free ratios 0.87 and 0.86 with upper bounds above 0.94, so the ceiling rule still fails there. Section 5.3 can say that a persistent trait and a per-episode multiplier match on early k, on the zero share, on the top-20% share and on final size once day-28 incidence is matched, and that the earlier 14.5% final-size gap was a property of matching on early R. Section 5.10 can treat the 585 chain as a hospital artefact: splitting hospitals, health centres and nursing homes, and leaving cinemas and churches aggregated, brings the longest chain to 150. The eight-week peak comparison should not be described as a 5.4% fall. On a run that contains the peak, the repaired file matches the current file.

## 10. Method

R3-1 reads the round-2 traces, the reassignment file and the presence file. No new PanSim runs. The five-seed probability groups chains in file order, five at a time, and compares the fraction with total size above 50 to the Dwass probability for five introductions. R3-2 permutes type-5 location ids with seed 20261014 and repairs within-agent duplicates by swaps that preserve venue counts. Calibration of day-28 infections uses the log-log secant, and a log-k interpolation when earlier rounds bracket the target. The volume cells and the two configuration-(b) event arms needed that interpolation after a secant with slope capped at 3 overshot. Hospital area is the sum of I5_h, I6_h and R_h. Wave-1 is the first 105 days and the wave-2 peak is the maximum from day 140. Early k uses index cases infected on days 1–10 with a known infector, and offspring within 14 days. Final size is cumulative infections at the end of the run (day 112 for R3-2, day 70 for R3-5). Cluster size is the number of trace infections at the event's location on the event's day. The non-empty median is over events with at least one infection.
