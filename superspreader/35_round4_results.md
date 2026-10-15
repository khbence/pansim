# ACT-049 note 35: round 4

15 October 2026. Instructions are note 34. Reads with notes 31 and 33. Nothing in `locations1.json`, `agents2.json`, `locations3.json` or `agents4.json` was edited.

## Deviations, and errors found before the runs

1. The simulator could not restrict events by capacity. A flag `--eventMaxCapacity` was added. Zero, the default, leaves eligibility and the random-number sequence unchanged. This is the restriction in the instruction, not the virtual-split fallback. A split would have created extra small rooms that still hold events.
2. `homeVenueFactor` was ignored whenever the venue profile was off, because the amplification block was skipped in that case. The block now also runs when a home weight is set. The default (home weight unset, profile off) is unchanged, so earlier runs are unaffected. W8 used the caregiving profile, so it was already inside the block. The first three shedding-shape pilots were identical for this reason and were discarded.
3. R4-3 uses the W9 protocol, coefficient set so early R is 1.3, levers from day 21, 12 weeks. The five existing lever arms were matched that way. Matching only the new arms on day-28 infections would have made the seven-arm table incomparable.
4. R4-1 uses the exact R3-5 coefficients, 0.000497841223604006 and 0.0004972587500573815.
5. The R4-2 day-28 target is the existing no-event mean, 6,211.25, not a new baseline run.
6. Factors 2,500 and 10,000 both gave early k below 0.3, so they did not bracket 0.3 and the failure branch (k still above 0.6 at 10,000) did not apply. Factors 600 and 1,200 were added, and 1,091 was interpolated between them.
7. Note 33 section 4.1 describes two different adjustments, dividing by the day-28 ratio and a regression on day-28 count. Both are reported. The regression slope at day 112 is 1.23 infections of final size per extra day-28 infection, not the 3.5 found at day 70.
8. A capacity filter does not keep a day's infections inside the design capacity. People turn over, so the cluster can exceed the cap. The observed maxima do.

## 1. Predictions

| task | prediction | result |
|---|---|---|
| R4-1 (i) | P0/E final-size ratio, 90% interval inside 0.90–1.10 | met. Raw 0.993–1.019. Adjusted for day 28, 0.993–1.012 |
| R4-1 (ii) | E and P0 end 2% to 8% below A0 at day 112 | met. Raw 7.1% and 6.6% below. Adjusted 6.7% and 6.5% below. Neither is more than 10% from A0 |
| R4-1 (iii) | peak day of E and P0 within 3 days of A0 | met for the means (A0 68.7, E 69.8, P0 68.2). One E replicate peaks on day 60 |
| R4-2 (i) | at capacity 100, early k 0.3 between factors 1,200 and 10,000 | the point estimate crosses 0.3 at factor 1,091. At 1,200, k is 0.266 and the replicate interval 0.22–0.34 still includes 0.3 |
| R4-2 (ii) | at that factor, non-empty median at most 20, 90th percentile at most 60, and no cluster above the capacity | failed. Median 29, 90th percentile 104, maximum 1,428 at a type-8 location |
| R4-3 | venue minus household positive for the venue-profile arm near share 0.424, and negative for the shedding-shape arm near 0.243 | the venue arm is positive, +10 points, interval 0.09–0.12. The shedding arm's point estimate is −0.8 points and its 90% interval, −0.03 to +0.02, includes zero |

The gap between E and A0 does not grow from day 70 to day 112. E is 9.5% below A0 at day 70 and 7.1% below at day 112. P0 is 4.8% below at day 70 and 6.6% below at day 112. P0/E falls from 1.051 at day 70 to 1.006 at day 112.

## 2. R4-0. Capacity on locations3.json

`35_capacity_by_type.csv` and `35_largest_locations.csv`. `121284_1803`, the largest location in the W3 chain of 202, is type 8 (long shop), index 75682, capacity 3,000.

Configuration (a) events use types 5 and 8 only.

| type | locations | at most 100 | at most 300 | above 300 | maximum |
|---|---:|---:|---:|---:|---:|
| 5 restaurant | 420 | 414 | 420 | 0 | 200 |
| 8 long shop | 1,280 | 1,047 | 1,216 | 64 | 10,000 |

Eleven type-8 locations have capacity above 1,000. A cap of 100 leaves 414 restaurants and 1,047 long shops eligible. A cap of 300 leaves every restaurant and 1,216 long shops. The capacity-3,000 shop is excluded by both.

## 3. R4-1. A0, E and P0 to day 112

Twelve new replicates, 16 weeks, coefficients from R3-5. Day numbers are the row numbers of the daily table, so day 28 is the same row as in round 3. Per-replicate summary: `35_r41_summary.csv`. Daily series: `35_r41_daily.csv`.

| arm | day 28 | day 70 | day 112 | peak prevalence | mean peak day |
|---|---:|---:|---:|---:|---:|
| A0 | 6,028 | 59,729 | 98,129 | 16,310 | 68.7 |
| E | 5,672 | 54,072 | 91,123 | 14,846 | 69.8 |
| P0 | 5,933 | 56,834 | 91,666 | 15,248 | 68.2 |

The new day-28 counts differ from R3-5 (6,221, 6,124, 6,299), as expected for new replicates. E is 5.9% below A0 at day 28 in this draw.

Regression of day-112 count on day-28 count with an arm term, evaluated at A0's day-28 mean of 6,028: A0 98,129, E 91,560, P0 91,783. The slope is 1.23. Adjusted ratios: P0/E 1.002 (90% interval 0.993–1.012), E/A0 0.933 (0.924–0.941), P0/A0 0.935 (0.930–0.942). Raw P0/E is 1.006 (0.993–1.019).

## 4. R4-2. Events at capped locations

Configuration (a), rate 0.3%, types 5 and 8, locations with capacity above the cap excluded. Day-28 infections matched to 6,211. Early k is the pooled index-case estimate, with a replicate-bootstrap interval. Clusters are events with at least one infection. `35_r42_events.csv`.

| cap | factor | day 28 | early k (interval) | non-empty median | 90th | maximum | max type |
|---:|---:|---:|---|---:|---:|---:|---:|
| 100 | 600 | 6,064 | 0.64 (0.53–0.82) | 20 | 82 | 1,312 | 8 |
| 100 | 1,091 | 6,194 | 0.31 (0.27–0.37) | 29 | 104 | 1,428 | 8 |
| 100 | 1,200 | 6,196 | 0.27 (0.22–0.34) | 27 | 96 | 296 | 5 |
| 100 | 2,500 | 6,093 | 0.13 (0.11–0.17) | 33 | 134 | 355 | 5 |
| 100 | 10,000 | 6,139 | 0.06 (0.05–0.08) | 42 | 161 | 421 | 5 |
| 300 | 2,500 | 6,203 | 0.14 (0.12–0.17) | 33 | 128 | 1,929 | 8 |
| 300 | 10,000 | 6,289 | 0.05 (0.04–0.07) | 41 | 162 | 2,822 | 8 |

Early k of 0.3 is reached, at a factor near 1,090 when the cap is 100. It is not reached inside the registered window as a point estimate: at factor 1,200, k is already 0.27, though that replicate interval still covers 0.3. Raising the factor further lowers k, to 0.06 at 10,000. The failure branch, k stuck above 0.6, did not occur.

At the factor where k is 0.31, the non-empty median is 29 and the 90th percentile is 104, and the largest cluster is 1,428. All three parts of the second prediction fail. The same pattern holds at capacity 300: the largest cluster is 1,929 at factor 2,500 and 2,822 at 10,000. Restricting the venue list does not keep a one-day cluster inside the design capacity.

## 5. R4-3. Mechanism against household share

The venue-profile arm is shedding shape 0, covid profile, strength 0.5, coefficient 0.0003. The two-seed pilot share was 0.414, inside 0.01 of 0.424. The twelve-replicate share is 0.413.

The shedding-shape arm is shape 0.3, venue profile off, home weight 0.423, coefficient 0.000528. Home weight is what was tuned: shape alone, at home weight 1, gave a share of 0.353, and home weights 0.35 and 0.15 gave 0.229 and 0.140. The twelve-replicate share is 0.250, inside 0.01 of 0.243.

| arm | household share | venue closure | household isolation | ventilation | venue minus household |
|---|---:|---:|---:|---:|---|
| venue profile, strength 0.5 | 0.413 | 57.0% | 46.9% | 51.8% | +10.1 points (0.09 to 0.12) |
| shape 0.3, home weight 0.42 | 0.250 | 47.1% | 47.9% | 44.1% | −0.8 points (−0.03 to +0.02) |

The venue-profile arm stays positive at a high household share, which is the family result from round 3. The shedding-shape arm at a low share does not keep the −11 point difference seen at shares 0.37–0.42. Its difference is indistinguishable from zero. Per-replicate final sizes and household shares for all seven arms are in `35_r43_lever_replicates.csv`.

## 6. Histograms and volume replicates

No new runs. `35_chain_size_histograms.csv` has every chain size from 1 to the longest, for the old map, the new map with hospitals unsplit, the hospital-only split, W3, W10 and W8. `35_r45_offspring_hist.csv` is the early-offspring histogram for A0, E and P0 from the R3-5 traces (index cases on days 1–10, offspring within 14 days). `35_r46_volume_replicates.csv` is day-28 infections and early k for each replicate of the four volume cells. The traces were still on disk.

## 7. Method

Day 28 is the 28th daily row. Final size in R4-1 is cumulative infections on day 112. The regression is ordinary least squares of that count on the day-28 count and indicators for E and P0, with A0 as the reference, evaluated at A0's mean day-28 count. Intervals resample the twelve replicates. Early k pools the index cases and the interval resamples replicates. Cluster size is the number of trace infections at the event location on the event day. The non-empty median is over events with at least one infection. Negative-binomial expectations are not recomputed in this note; the chain histogram is the input for that figure.
