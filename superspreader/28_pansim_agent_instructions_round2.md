# ACT-049 note 28: instructions for the simulation-running agent, round 2

13 October 2026. These follow `26_venue_file_results.md` and my reading of it in `27_reading_of_26_venue_file_results.md`. Read 27 first; it says why each item is here. Everything is written to be run without further input. Where a decision belongs to the project lead I say so and stop.

## 0a. Corrections to notes 27 and 28, 13 October 2026

The simulation agent checked notes 27 and 28 before starting and found four errors. All four are confirmed and are corrected here. Its recomputations of everything else in note 27 reproduce my figures.

1. **The old map's longest chain was not "ordinary".** Note 27 gives P(any chain at least 254) = 0.35 on the old map. That number was computed with the *new* arm's mean and $k$ (0.397, 0.030), which does give 0.348. At the old arm's own parameters (0.363, 0.034) over 38,545 chains it is **0.062**. So a 254-case chain on the old map was uncommon, about 1 study in 16, and a 585-case chain on the new map is about 1 in 2,000. The contrast is weaker than note 27 claimed and the conclusion survives: the new map has an extreme the benchmark does not allow, and 0.062 suggests the old map had a mild excess too, which is consistent with cinemas being aggregated on both maps. Note 27's sentence that the old maximum "was ordinary", and the same phrase in section 0 below, are withdrawn.
2. **The busiest restaurant's fill is 7.8, not 3.1.** It is a cafe of design capacity 40 with a peak of 310 (3,996 assigned agents); the venue with 4,022 assigned is a different capacity-40 cafe. Note 27 and the W10 step 0 text divided 310 by 100. Where the fill of the busiest venue matters, use 7.8.
3. **The W10 feasibility test is infeasible at a target fill of 1.0, and note 28 pointed the wrong way.** See the rewritten step 2 below. In short: the capacity mix is 190 venues at 40, 210 at 100, 14 at 80, 3 at 120 and 3 at 200, a total of 30,680 and a mean of 73, not 100. Feasibility *improves* as the target fill rises, so the quantity to report is the **smallest** feasible fill, not the largest, and it is about 1.10.
4. **Note 27 was wrong that note 26 gave no reason for V6, V7 and V8.** Note 26 states that they were later in the order and that V7 is blocked because area is still zero for types 5, 9 and 10. Withdrawn.

Two test definitions in note 28 were under-specified and are fixed in place below: the W1 threshold on a 585-case chain, and the gap between W3's two registered branches.

## 0. State of play in four lines

1. The new map leaves the aggregate fit, the type shares and the person-level dispersion results unchanged (V0, V1, V2 on $k$). Nothing in this round needs those redone.
2. The Nipah chain match holds at thresholds 10 and 50, but the longest chain is 585, which a negative binomial at the arm's own mean (0.397) and $k$ (0.030) allows with probability about 0.0005 over 38,613 chains. The old map's longest chain (254) had probability 0.062 at its own parameters, which is uncommon but not of the same class. This is the main open item.
3. The thin middle of the caregiving chains survived both the hospital and the restaurant split. Its cause is not identified. Two explanations are open and one test separates them at no GPU cost.
4. The V2 final-size gap is early-R drift. A final size within 5% needs two arms matched on early $R$ to about 0.012, so the calibration has to be tightened.
5. The agent-to-restaurant assignment is repaired in W10, at a target peak fill of 1.10 decided by the project lead on 13 October 2026. No venues are added this round. Churches still need about 172 more locations and are left to the virtual split of W3.

## 1. Rules for everything below

- **Pre-registered predictions are fixed before you look.** Each task states its predictions and what each outcome means. Report a failed prediction as failed. If you change a test rule after seeing data, label the new rule post hoc and report the original rule's result beside it. Note 26 did this well for V3; keep doing it.
- **State the threshold definition.** Write "above 50" or "at least 50" explicitly, in the table and in the code. The two differ by about 4% in expected counts here.
- **Use exact expected counts for negative-binomial benchmarks.** Do not simulate them. The total size $n$ of a chain from $i$ introductions, for offspring negative binomial with mean $R$ and dispersion $k$, has probability $(i/n)\,P(S_n = n - i)$, where $S_n$ is negative binomial with shape $nk$ and success probability $p = k/(k+R)$. In Python:

```python
import numpy as np
from scipy.stats import nbinom
def p_total_gt(R, k, i, cap):          # P(total chain size > cap) from i introductions
    p = k/(k+R); n = np.arange(i, cap+1)
    return 1 - ((i/n)*nbinom.pmf(n-i, n*k, p)).sum()
# expected chains above 50 among N single-introduction chains: N*p_total_gt(R, k, 1, 50)
# five-seed pooled probability of total above 50: p_total_gt(R, k, 5, 50)
```

  If scipy is unavailable, implement the same formula with log-gamma terms: $\log P(S_n = n-i) = \log\Gamma(nk+n-i) - \log\Gamma(nk) - \log\Gamma(n-i+1) + nk\log p + (n-i)\log(1-p)$. The agent has confirmed this reproduces the scipy values.

  Use the arm's own realised mean offspring and $k$, not the calibration target and not the literature pair, unless the table is explicitly a comparison with the literature. Note 26 compared the arm's five-seed 1.2% with 0.61% at the literature pair (mean 0.33) when the arm's mean was 0.397; the right comparator is 1.38%.
- **Intervals** are cluster bootstraps over replicates (and over events where events are involved), never over infectors.
- **Early $R$ in every ratio** is measured on the same replicates as the numerator.
- **Do not change the input location or agent files.** If a task seems to need a different map, build the variant as a virtual split in code (the X1 wards recipe: split by assigned agents, preserve total capacity and each agent's assignment) and say so. Anything that needs a repaired file goes to the project lead (W9).
- **Outputs.** One note, `29_round2_results.md`, plus the tables as CSV files with the prefix `29_`. Push to the shared Drive folder. If the push fails again, save the commit hash and a copy of the note and CSVs in the repo and say so first thing in the note.

## 2. Tier 1: no GPU, from traces already on disk. Do these first.

### W1. Empirical-offspring branching test (separates two explanations of the thin middle)

**Why.** The V5 benchmark is a negative binomial with the arm's mean and $k$. If an arm's offspring distribution is not negative binomial, chains differ from the benchmark even when offspring are independent. In the caregiving arm homes have weight 2.5 against 0.35 for most other types, and households have a mean size of about 2.6, so household-dominated transmission may cap offspring and thin the 10-to-50 range without any correlation across generations.

**Do.** For each of these arms on the new map: the V4 shape-0.03 arm, the V5 flat arm, the V5 caregiving arm (hospitals, health centres and restaurants split), and the X1 caregiving arm (restaurants aggregated, from the old trace): take every infected agent's offspring count from the trace (every node, including those with none) and run a branching process in which each case draws its offspring count independently from that arm's own empirical distribution. Use at least 100,000 chains per arm with a cap of 5,000 on chain size. Report, for the empirical-iid process, the negative binomial and the observed arm: chains above 10, above 50, above 100, above 200, and the longest chain, with 95% Poisson bands for the observed counts.

**Predictions, registered now.**
- V5 caregiving arm: if the empirical-iid process reproduces the observed chains above 10 (220) within its Poisson band, the thin middle is a property of the offspring distribution's shape and not of correlation between generations, and the paper's phrase "setting-correlated offspring" must go. If it lands near the negative binomial's 313, the offspring are correlated across generations. My expectation, which is an inference and not a result, is that it lands closer to 220 than to 313.
- V5 flat arm: empirical-iid, negative binomial and observed should agree (485, 499).
- V4 arm: the threshold applies to a study of the same size as the observed one, so compute P(at least one chain of 585 or more among 38,613 chains) under the empirical-iid process, not the maximum over however many chains you simulate. Simulate in blocks of 38,613 and report the share of blocks containing such a chain, over at least 300 blocks. If that share exceeds 0.05, the 585 is explained by the shape of the offspring tail and W3 becomes a refinement. If it is well below 0.05, the 585 reflects correlation across generations, and W2 and W3 are decisive. Report the same quantity for the negative-binomial benchmark (0.00052) and for the old-map arm at its own parameters (0.062 at threshold 254).

### W2. Composition of the long chains in the V4 arm

**Do.** From the V4 trace (40 replicates, 38,613 chains): list the 20 longest chains with their size, number of generations and the location type of each infection by generation. For chains above 200 and separately for chains of 51 to 200, report the share of infections by location type (restaurant, church, park, cinema, shops, work, school, hospital, home, other) and, for each chain, the number of distinct location ids and the largest share of the chain's infections in any single location. Report the same for the old-map V4 trace if the old trace is still on disk. Also report chains above 100, above 200 and above 500.

**Predictions, registered now.** If the cinemas drive the tail, then among chains above 200 the share of infections in type 6 is at least 25% (against 4.6% of infections overall in V1), and in the 585-case chain one cinema holds at least 30% of the chain's infections. If a church or park drives it, the same threshold applies to that type. If none of the three applies, say which types do carry the chain; do not name a cause the data do not show.

### W4a. The binned offspring curve, new map (the reference-free V3)

**Why.** The pre-registered ceiling rule predicts top-1% offspring from the lower-half slope, which is 1.5 times the early reproduction number at shape 0.3 and about 2 to 2.5 times at shapes 0.08 and 0.03. Offspring per unit multiplier already falls across the bulk, so the rule measures concavity against an inflated reference. See note 27 section 4.

**Do.** On the existing new-map traces at shapes 0.3, 0.08 and 0.03 (the V3 runs), with the same index-case definition (infected on days 1 to 10 with a known infector, offspring within 14 days): report a table with a zero-multiplier bin and 25 log-spaced bins for the rest, giving the number of infectors, mean multiplier, mean offspring, offspring per unit multiplier, and the replicate-bootstrap interval of the last. Also report, per shape, the measured early $R$, the measured top-1% mean multiplier, the observed top-1% mean offspring, and the ratio $\text{observed} / (R \times \text{top-1\% mean multiplier})$ with its bootstrap interval. Keep the original lower-half and moderate-bin ratios in the same table.

**Predictions, registered now.** The reference-free ratio is at least 0.80 at all three shapes on the new map, so a venue ceiling cannot be claimed at any shape. A ceiling claim for a given shape needs the ratio below 0.75 with its bootstrap upper bound below 0.85. State plainly whether it is met.

## 3. Tier 2: short runs

### W3. Split the remaining aggregates and repeat the Nipah arm (the decisive test for 5.10)

**Do.** Build a virtual variant of the new map in which type 6 (cinemas) and type 9 (churches) are split into rooms by assigned agents, preserving total capacity and each agent's assignment: target median peak occupancy about 150 for cinemas (inside the 100 to 1,000 label) and about 300 for churches (inside 50 to 500). Leave restaurants, parks and hospitals as in note 26's V5 map (hospitals and health centres in wards of about 30). Run the V4 arm on it with the same protocol (shape 0.03, flat multiplier, venue profile and events off, 40 replicates, coefficient calibrated to mean offspring 0.33), and report the table of note 26 section V4 plus chains above 100, above 200 and above 500, with exact negative-binomial expectations at the realised mean and $k$.

**Predictions, registered now.**
- The longest chain is at most 300 and chains above 100 fall inside the Poisson band of the exact expectation. Then the 585-case chain was an aggregation artefact of cinemas and churches, and the Nipah match is clean on a fully realistic venue map.
- If the longest chain stays at 400 or more, the tail does not come from types 6 and 9; W2's composition result says where to look next.
- If the longest chain lands between 300 and 400, neither branch applies: report P(at least one chain of that size or more among the arm's own number of chains) at the arm's realised mean and $k$, and treat the test as passed if that probability exceeds 0.05, unresolved otherwise.
- Either way report realised mean and $k$, and, because the earlier arm ran at mean 0.363 on the old map and 0.397 on the new one, also report the five-seed pooled probability against its own exact expectation. Do not compare with the literature pair unless the means match.

### W4b. Old-map comparators for the offspring curve

**Do.** On the old map (`locations0.json`, `agents` as used in PS1), run the V3 arms at shapes 0.08 and 0.03, flat uncoupled multiplier, eight replicates, calibrated to early $R$ 1.39 with the W5 second-stage calibration, and apply the W4a analysis to them and to the existing PS1 trace of the shape-0.3 arm. **Prediction, registered now:** the reference-free ratio on the old map at shape 0.3 is about 0.76 (from D1: 11.8 against 1.333 times 11.6), and the old-map ratios at 0.08 and 0.03 are within 0.1 of the new-map ones. If the old-map ratios are clearly lower at every shape, with intervals that do not overlap the new-map ones, then part of the original tail bend came from the aggregated venues, which would revise paper Section 5.5.

### W5. Match the early reproduction number before comparing final sizes (V2b)

**Why.** Final size changes by a factor of about $e^{4.2\,\Delta R}$ across the three old arms, and the two-seed calibration of a heavy-tailed arm missed the twelve-replicate $R$ by 0.12 to 0.14. See note 27 section 5.

**Do.** For arms A0, E and P0 on the new map: after the two-seed calibration, run 12 replicates, take the 12-replicate early $R$, and correct the coefficient by the log-log secant. Repeat until the 12-replicate early $R$ is within 0.03 of 1.39 for each arm and E and P0 are within 0.02 of each other, or stop after three rounds. If three rounds do not suffice, run each arm at three coefficients spanning early $R$ of 1.30 to 1.48 and report final size interpolated at 1.39 by regressing $\ln(\text{final size})$ on early $R$. Report the seed-to-seed standard deviation of early $R$ for each arm, which is the reason the two-seed calibration is unreliable.

**Predictions, registered now.** At matched $R$, E and P0 final sizes are within 10% of each other (within 5% on the old map), and early $k$ within 13%. If they differ by more than 15% at matched $R$, then the observational equivalence of a persistent trait and a per-episode multiplier does not extend to the epidemic's size, and that would be a new finding for paper Section 5.3.

### W6. Audit extension and the V1 hold-out (the audit the note did not include)

**Do.** (a) For types 5, 6, 7, 8, 9, 10, 12, 14 and 22, report the median, 90th percentile and maximum of peak occupancy over capacity, and the share of locations with peak above capacity, in the format of the phase 3 table. (b) Report the percentiles (5th, 25th, 50th, 75th, 95th, 99th, maximum) of assigned agents per location for the same types, and the correlation between assigned agents and capacity. (c) Complete V1: report the wave-1 and wave-2 hospital areas on the new map beside the old, with the minimum detectable difference (Welch, 80% power) next to each verdict. If four replicates are too few for a useful minimum detectable difference, run eight 37-week replicates. Do not refit the variant vector unless wave 2 differs from the old baseline by more than that minimum detectable difference.

**Prediction, registered now, and already met on existing data.** I predicted a median restaurant fill between 0.5 and 1.0 with a 90th percentile above 2. From the existing three-week occupancy file the agent reports median 0.93, 90th percentile 3.25, and 47% of type-5 locations peaking above design capacity. The prediction is met, and the fill distribution is dominated by assignment imbalance rather than venue size. Also reported: church median fill 3.7 with 98% above capacity, every cinema above capacity (median 1.4), and park median 0.39. Treat W6(a) as answered for the current map, reporting it as a measurement rather than a new run; it still needs repeating on the repaired map (W10 step 5 item 3).

## 4. Tier 3: after Tier 1 and 2

### W7. Events, now that restaurants are realistic (V6 of note 25)

**Why.** This is the highest-upside item for the setting side of the paper. The per-infector mean event offspring is occupancy-independent, so the restaurant map should leave event-driven individual dispersion roughly where E5 found it while making event clusters realistic.

**Do.** Two configurations: (a) events eligible only at restaurants (type 5) and long-visit shops (type 8); (b) events at all eligible types including cinemas and churches. For each, at event rate and factor of (0.3%, 1000x), (1%, 200x), (0.3%, 100x) and (1%, 20x): calibrate the coefficient to early $R$ 1.39 by bisection using four-replicate pilots, then run 12 replicates. Log every event: type, location id, day, number present, number of infectors present, infections caused. Report early $k$ (including event infections), the share of infections from events, and the distribution of event cluster sizes (median, 90th percentile, maximum) with intervals that resample events and replicates. Also find the (rate, factor) at which early $k$ reaches 0.3 in configuration (a) and report its event cluster-size distribution beside the Skagit Valley rehearsal (61 attendees, 53 confirmed or strongly suspected cases).

**Predictions, registered now.** In configuration (a) at (0.3%, 1000x) early $k$ is within a factor of 2 of the old E5 value of 0.076 (so between 0.04 and 0.15), and the median event cluster is at most 100. In configuration (b) the largest clusters come from type 6 or 9. If $k$ rises above 0.15 in (a), then the old E5 result was manufactured by the aggregates after all and my kernel argument in note 25 section 2 is wrong; say so.

### W8. Thin-middle follow-up (run only if W1 shows correlation across generations)

If W1 reproduces the observed deficit with independent offspring, do nothing here. If it does not, run the V5 caregiving arm with the home base infectiousness lowered from 2.5 to 0.35 and everything else unchanged, 38,500 chains. **Prediction:** if household truncation drives the deficit, observed chains above 10 move to within 15% of the negative binomial (from 30% below). If the deficit is unchanged, it comes from hospitals, nursing homes or churches, and the next step is the same location-id composition analysis as W2 on the caregiving trace.

### W10. Rebalance the agent-to-restaurant assignment (new file)

The project lead has asked for this, so it is no longer gated. Read the arithmetic in step 0 before writing any code, because the cap I suggested in note 27 was wrong and following it would make the file worse.

**Step 0. The target is peak occupancy, not assigned count.** Note 27 suggested capping assigned agents at about three times design capacity. That is infeasible and it is the wrong quantity. There are 153,710 agents with three type-5 slots each, so 461,130 assignments across 420 restaurants, a mean of 1,098 per venue. A cap of three times a design capacity of 100 is 300, which would leave about two thirds of the assignments homeless. Discard that rule.

The quantity that matters is how many people are in a venue at once, relative to what the venue holds. Only about 7% of a restaurant's assigned agents are present at its peak: the median venue has 944 assigned and a peak of 69. Define

$$\text{presence rate } \pi = \frac{\text{peak occupancy}}{\text{assigned agents}}, \qquad \text{assignment cap} = \frac{\text{design capacity} \times \text{target fill}}{\pi}.$$

At $\pi = 0.073$ and a design capacity of 100, a target peak fill of 1.0 is an assignment cap of about 1,368. That is above the mean of 1,098, so the cap binds only on the crowded venues, which is the intent. The busiest restaurant by occupancy is a capacity-40 cafe with 3,996 assigned and a peak of 310, a fill of 7.8; a separate capacity-40 cafe holds 4,022 assigned agents. Capping either at the capacity-40 quota of about 547 would bring its peak to about 40.

**Step 1. Measure, do not assume.** From one short uncontrolled reference run with `--occupancyStats` on the current map, report per type-5 location: assigned agents, 95th percentile of daily peak occupancy, design capacity, $\pi$, and fill. Report the distribution of $\pi$ (5th, 50th, 95th percentile) and the count of venues at each distinct design capacity. My $\pi = 0.073$ is a ratio of medians from note 26 and is not a per-venue measurement.

**Step 2. Feasibility, already computed: a target fill of 1.0 is not reachable.** The map can absorb the assignments at target fill $\phi$ only if

$$\sum_{\text{venues}} \frac{\text{capacity}_v \, \phi}{\pi_v} \ \ge \ 461{,}130 .$$

Feasibility improves as $\phi$ rises, so the quantity to report is the **smallest** feasible $\phi$. Note 28 version 2 said "largest", which pointed the wrong way; that is corrected here.

The agent has measured the capacity mix and it is not uniform: 190 venues at capacity 40 (45%), 210 at 100, 14 at 80, 3 at 120 and 3 at 200. Total capacity is 30,680, a mean of 73. At $\pi = 0.073$ the map absorbs about 420,300 assignments at $\phi = 1.0$, which is 91% of the 461,130 needed, so **the smallest feasible target fill is about 1.10**. My cafe rule in version 2 (infeasible above about a 33% cafe share) correctly flagged this; the version 2 feasibility CSV asserted "rebalancing alone is enough" on the wrong assumption that every venue has capacity 100, and that row is withdrawn.

**Decision taken by the project lead, 13 October 2026: run the repair at a target fill of 1.10.** No venues will be added for this round. The per-venue assignment caps are therefore $1.10 \times \text{capacity}_v / \pi_v$, which at the pooled $\pi = 0.073$ are:

| design capacity | venues | assignment cap |
|---:|---:|---:|
| 40 | 190 | 603 |
| 80 | 14 | 1,205 |
| 100 | 210 | 1,507 |
| 120 | 3 | 1,808 |
| 200 | 3 | 3,014 |

**Read this before solving: 1.10 is a knife edge.** Total absorbable capacity at that fill is about 462,300 against 461,130 assignments, which is 99.7% utilisation and an aggregate slack of about 1,200 assignments, or 0.3%. Two consequences follow and both need handling rather than discovering.

| target fill | absorbable | utilisation | aggregate slack |
|---:|---:|---:|---:|
| 1.00 | 420,300 | 110% (infeasible) | -40,900 |
| 1.10 | 462,300 | 99.7% | +1,200 (0.3%) |
| 1.15 | 483,300 | 95.4% | +22,200 (4.8%) |
| 1.20 | 504,300 | 91.4% | +43,200 (9.4%) |

1. **The distance objective has almost no room.** At 99.7% utilisation nearly every venue must be filled to its cap, so the cap behaves like an equality quota and minimising distance can only choose which agents go where, not how full a venue is. Expect substantial displacement, and expect the nearest-venue structure of the current file to be largely rewritten rather than lightly adjusted. The first value with real slack is 1.15.
2. **A 0.3% error in the presence rate flips it to infeasible.** The pooled $\pi = 0.073$ is my ratio of medians from note 26, not a measurement. Fill 1.10 becomes exactly infeasible at $\pi = 0.0732$. Step 1 measures $\pi$ per venue, so the feasibility sum must be recomputed with the measured values before solving.

**Escalation rule, pre-authorised so that no round trip is needed.** Compute $\sum_v \text{capacity}_v \phi / \pi_v$ with the measured per-venue $\pi_v$ at $\phi = 1.10$.

- If it exceeds 461,130 by at least 2%, solve at 1.10 as decided.
- If it does not, or if the solve leaves the 95th percentile of displacement above 5 km, raise $\phi$ in steps of 0.05 to the first value that clears both, solve there, and report the chosen $\phi$, the feasibility sum at each step tried, and the displacement distribution at both 1.10 and the chosen value. Do not go above 1.25 without asking.
- Where a venue has too few assigned agents for a stable $\pi_v$ (fewer than 50 assigned, or fewer than five days with any occupancy), use the pooled $\pi$ for that venue and say how many venues this affected.

Report the realised peak fill distribution either way; the validation table below is written for $\phi = 1.10$ and scales with $\phi$.

**Step 3. Reassign.** Keep every agent's three type-5 slots, keep the slots distinct, and keep the assignment spatially sensible, because the current file assigns each agent to its nearest distinct places and that structure should survive. This is a transportation problem: minimise total assigned distance subject to a per-venue quota equal to the step 2 cap. Two acceptable implementations:

- **Min-cost flow**, agents to venues with quotas, cost equal to distance. Exact and the preferred option if a solver is available (`scipy.sparse.csgraph.min_weight_full_bipartite_matching` will not take quotas directly; `ortools` min-cost flow will, as will a simple auction or successive-shortest-path implementation).
- **Greedy with displacement**, if no solver is available: process agents in a random fixed-seed order, assign each to its nearest venue with quota remaining, and at the end report the distance displacement distribution. Record the random seed in the file header.

Rules that are not negotiable: the number of slots per agent per type is unchanged; no agent gets the same venue twice; the seven restaurants with no assigned agent either receive agents or are reported as genuinely unreachable; nothing about time use, schedules, or any other location type changes.

**Step 4. Validate.** Report, old file against new file:

| check | pass condition |
|---|---|
| assignments conserved | exactly 461,130, and three distinct type-5 slots for every one of the 153,710 agents |
| peak fill distribution | median between 0.5 and 1.1, 95th percentile at or below 1.3, maximum at or below 1.6 (at a target fill of 1.10; scale with $\phi$ if the lead changes it) |
| assignment spread | maximum over median at or below 2 (now 4.3; the raw spread is 5 to 4,022) |
| displacement | median increase in assigned distance and the 95th percentile, in metres, plus the share of agents whose nearest-venue assignment changed. Expect this to be large at 99.7% utilisation; it is a measurement, not a failure, unless the 95th percentile exceeds 5 km, which triggers the escalation rule |
| empty venues | zero, or a list with the reason |
| time use unchanged | minutes per agent per type identical to the old file, to the byte |
| regression | a run on the new file with all heterogeneity flags off reproduces the old file's early $R$ to within the two-seed calibration tolerance, and types other than 5 have byte-identical location records |

**Step 5. Re-run after the repair, with these predictions registered now.** Only four things need re-running, and each prediction says what a failure would mean.

1. **V1 invariance.** Eight replicates, baseline. Prediction: the infection coefficient still accepts at early $R$ 1.39 unchanged, the restaurant share of infections stays within 0.5 points of 4.28%, and peak prevalence moves by less than 2%. A failure would mean the rebalancing changed transmission and not just co-presence, which should not happen under the frequency-dependent kernel, and would need explaining before anything else is believed.
2. **W3 repeat, the Nipah arm.** On the repaired map with cinemas and churches split as in W3. Prediction: the longest chain falls below the W3 value, because the busiest restaurant's venue-day co-presence falls from 310 to about 100. If W3 already brought the longest chain below 300, this should bring it below 250.
3. **W6 fill audit.** Prediction: the restaurant rows now pass the phase 3 criterion of no more than 1% of occupied locations with peak above capacity.
4. **V7, crowding, which this repair unblocks.** After the three checks above pass, run the volume-model cells E1 (non-residential, mid air exchange), the low and high air-exchange sensitivities, and the fill-pinned control. Prediction: the fill-pinned and full-volume cells now differ by more than their intervals, which they did not on the peak-anchored capacity file (2.86, 2.78 to 3.82 against 3.24). That would establish crowding as a mechanism in its own right rather than a reparameterisation of the venue profile. Note that area is still zero for types 5, 9 and 10, so the type factors must come from the capacity-times-area-per-person table of note 12 rather than from the file.

**Step 6. Churches cannot be fixed this way, and parks need nothing.** The same arithmetic, applied to the other two changed types, gives:

| type | assignments | venues now | peak if perfectly balanced | fill if balanced | venues needed for fill 1.0 |
|---|---:|---:|---:|---:|---:|
| 5 restaurant | 461,130 | 420 | about 80 | 0.80 | 337 |
| 9 church | 489,084 | 52 | about 645 | 4.30 | about 224 |
| 10 park | 500,574 | 167 | about 132 | 0.44 | 73 |

Restaurants are solvable by rebalancing alone: perfectly balanced they sit at a peak of 80 against a design capacity of 100 and a type label of 20 to 100. Parks already have more than twice the venues they need. **Churches do not: even a perfect rebalancing leaves a peak of about 645, above both the design capacity of 150 and the type label of 50 to 500, so type 9 needs about 170 more locations, not a better assignment.** Report that deficit to the project lead and do not invent church locations. Until they exist, keep using the virtual church split of W3 for any experiment that touches the tail, and say in every table which map was used. Cinemas are unchanged at 20 locations and remain the largest aggregate in the file.

**Step 7. File hygiene.** Write `locations3.json` and `agents4.json` (do not overwrite `locations1.json` or `agents2.json`, and do not reuse the numbering of any file the lead created). Ship the agent-to-venue mapping as `29_type5_reassignment.csv` with columns `agent_id, old_location_ids, new_location_ids, old_distance_m, new_distance_m`, the solver and seed in the note, and the step 1, 2 and 4 tables. State in the note's first section which experiments in this round used which map.

### W9. Levers, and what remains gated

- **Levers (V8).** Run the three household-share arms of X2 (shares near 0.25, 0.37 and 0.45) on the new map, measuring the household share in the same runs, plus the ventilation lever, with 12 replicates and levers from day 21. Predict that the sign of venue closure minus household isolation changes between 0.37 and 0.39, and that the closure benefit at a matched share moves by less than 5 points from X2. Try to reach a share above 0.39, which the X2 series did not.
- **Crowding (V7).** Now handled in W10 step 5 item 4, after the assignment repair. Do not run it on the current map: with the fill distribution dominated by assignment imbalance it would measure note 12's capacity-error mechanism rather than crowding.
- **Still gated on the project lead.** The 170 additional church locations of W10 step 6, and any decision to add cinemas. Report the deficits; do not generate locations.

## 5. What not to do

- Do not re-run V1, V2's $k$ comparison, the coupled arms, the depletion check, the lognormal recalibration or the infectiousness-profile experiment. None of them involves the venue file.
- Do not add the distinct-contact field to the trace. Note 25 asked for it; I withdraw the request, because the episode budget (about 1,600) exceeds tail offspring (16 to 66) by 24 to 100 times and cannot bind on mean offspring.
- Do not describe the thin middle as caused by "the schedule", or the 585-case chain as caused by the cinemas, until W1 to W3 have tested it.

## 6. Order and what each could overturn

| order | task | cost | could overturn |
|---|---|---|---|
| 1 | W1, W2, W4a, W6 audit | no GPU | the paper's setting-correlated offspring claim (W1); the interpretation of the 585 chain (W2); the venue-day ceiling reading of V3 (W4a) |
| 2 | W5 | 3 calibration rounds of 12 replicates | the final-size half of paper Section 5.3 |
| 3 | W3 | one 40-replicate arm on a virtual map | the cleanliness of the Nipah match in paper Section 5.10 |
| 4 | W4b | two small arms | whether the map changed the tail bend at all |
| 5 | W6(c) | up to eight 37-week replicates | the hospital fit claim |
| 6 | W7 | the largest block | the paper's statement that a Skagit-type event is not representable |
| 7 | W8 | conditional | the mechanism of the thin middle |
| 8 | W10 (assignment repair; steps 0 to 4 need no GPU, step 5 does) | one solver run plus four short runs | unblocks crowding (V7); tightens the venue-day co-presence ceiling |
| 9 | W9 | partly gated | none; confirmatory |

## 7. Deliverables

`29_round2_results.md` with one section per task and a first section listing every pre-registered prediction as met, failed or not tested; `29_chain_counts.csv` (W1, W3); `29_long_chain_composition.csv` (W2); `29_offspring_curve.csv` (W4); `29_r_matched_final_size.csv` (W5); `29_audit_fill_and_assignment.csv` (W6); `29_events.csv` (W7); `29_type5_reassignment.csv` and the step 1, 2 and 4 tables (W10). If a task cannot be run, say which and why, and do not substitute another.
