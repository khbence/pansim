# ACT-049 results: where PanSim overdispersion comes from

The question is the one in `04_pansim_experiments.md`: how much of the simulated offspring distribution is biological shedding, how much is who meets whom, and how much is the place, and does that split change what an intervention does? The window is the first two Hungarian waves in `viz/fitted_parameters2.json`, from 23 September 2020 (day 0) through the alpha wave, which ends on 2 June 2021 (day 252). A 37-week job finishes a few days after that. Later variants are seeded on or after day 339, so they are outside this study.

## Answer

The fitted model, with the real closure file and quarantine policy 3, and with no extra heterogeneity, produces offspring dispersion k of about 2.6. About 37% of infectors account for 80% of secondary infections. Published SARS-CoV-2 estimates are usually below 1, and often near 0.1, which would put about 10% of infectors behind 80% of infections.

Schedules plus a modest venue-type profile do not produce that published tail. With no closures, real schedules give k of about 4.6. Reshuffling venues inside each location type, which keeps the occupancy distribution and breaks households, leaves k at about 4.4. The covid venue profile at full strength lowers k only to about 2.6. A per-infection gamma multiplier does copy its shape into offspring k, including the check that shape 0.1 leaves about 10% of infectors causing 80% of infections, and it does not by itself pile infections into a minority of venues. Because the median draw is far below the mean of 1, the same multiplier shrinks the epidemic. Dropping it, or the modest venue profile, onto the infection coefficient that was already calibrated without them moves the hospital curve off the fit.

Stronger amplification can cross k = 1. Venue-profile strength 4 gives k = 0.85, and the busiest 10% of infected venues account for 79% of infections. Strength 8 gives k = 0.54 and 84% in the busiest tenth of venues. Neither reaches 0.1. Both infect essentially the whole simulated population within eight uncontrolled weeks.

Detection changes k, and by less than an order of magnitude. Backward tracing inflates the mean offspring of the people it finds. It does not manufacture k near 0.1 out of a simulated k near 4.

Two mechanisms can share an eight-week k near 0.8 and an eight-week mean offspring near 1, and still differ by a factor of about three in first-week reproduction and by a factor of about 70 in prevalence on day 14. In both, cutting community hazard reduces infections more than isolating diagnosed people. The ranking in the proposal, with biology favouring isolation and settings favouring venue controls, is not what these runs do.

At a matched generation-1 reproduction number near 0.25, a caregiving profile lengthens the largest chain from 5 to 13 and moves k from Poisson to about 0.7. It does not raise the share of cases with zero offspring. No chain in the calibrated grid reached 50 cases. The runs use Szeged schedules and the PanSim natural history. They are not a Nipah forecast and not an H5N1 forecast.

## Instrumentation

The stack is off unless a flag turns it on, so a fitted run that does not pass the new flags is unchanged. The 37-week baseline below is that check: its hospital peaks sit on the scaled reconstruction.

- `--sheddingShape` draws a gamma multiplier with mean 1 once per infection episode. A value of 0 or less leaves the multiplier at 1. The draw is not kept for life, and it is not a distinct supershedder class.
- `--shuffleSchedules` permutes place assignments inside each location type. The multiset of venue occupancies stays the same. Co-membership, including households, is broken. Each agent is then placed in their new home.
- `--venueProfile off|covid|caregiving` and `--venueStrength` scale location infectiousness. Strength s applies `1 + s * (base - 1)`, and only where the input infectiousness is already positive. Public locations in `locations0.json` have infectiousness 0, so the profile does not turn them on.
- `--communityScale` multiplies infectiousness again for every type other than public, home, hospital, and nursing.
- `--eventRate` and `--eventFactor` give each restaurant, large-evening, long-visit, and church location an independent daily chance of transmitting at `eventFactor` for that day.
- `--infectionTrace` writes one row per infection. The kernel is still the frequency-dependent hazard, not a pairwise contact. The recorded infector is one weighted draw among the other people present, with weights equal to infectiousness.

Covid profile bases: home 1, school 1.4, classroom 1.6, work and shift work 1.1, restaurant 4, large evening 3, short shop 0.8, long visit 2.5, church 3, outdoor recreation 0.3, prison 2, hospital 1.8, health centre 1.3, nursing 2.2, university 1.5. Caregiving bases: home 2.5, church 3, prison 2, hospital 4, health centre 2, nursing 3, and 0.35 for every other type that already has positive infectiousness.

The infection probability in a step is `1 - exp(-k * density * locationInfectiousness * eventBoost * seasonality * dt)`, and density is the sum of agent infectiousness divided by the number of people present. A crowded venue is diluted by the same occupancy that an event boost is trying to amplify. That is why a small daily chance of a moderate event barely moves k.

## Design common to the runs

Inputs are the tuned progression `inputConfigFiles/progressions_Jun17_tune/transition_config.json`, `agentTypes_3.json`, `inputRealExample/agents1.json`, and `locations0.json`. Variant multipliers and progression scalings are the fitted vectors: infectiousness 0.98, 1.85, 2.06, 3.2, 3.41667, 4.55, 6.8 and progression 0.94, 1.03, 0.84, 0.72, 0.8208, 0.463, 0.45. The baseline infection coefficient is 0.00041.

About 178,750 agents are active. The reconstruction population used for scaling in `viz/` is 9,967,304, so one simulated person stands for about 55.8 people nationally. Official hospital peaks of 8,045 and 12,553 scale to about 144 and 225. Reconstruction infected peaks of 166,157 and 220,698 scale to about 2,980 and 3,960.

`-r` draws the initial state from `configRandom.json`, which places a few hundred infections and a small recovered fraction. Experiment 4 does not use `-r`. It seeds about five infections on day 0 with `superspreader/study/closures_seed.json` (`ExposeToMutation`, variant 0, fraction 0.000027). Uncontrolled runs use `closures_none.json`, whose rule list is empty. The empty file is required: `--enableClosures 0` still applies mutation seeding, masks, curfews, holidays, and school rules.

Quarantine policy 0 leaves people in place, 1 quarantines the diagnosed agent, 2 adds the household, and 3 adds classroom and workplace. Testing still diagnoses people when the quarantine policy is 0.

For epidemics that are still growing, k is the method of moments on index cases infected at least 14 days before the last infection in that trace: `k = mean^2 / (variance - mean)`. If the variance is at or below the mean, k is infinite, which means Poisson or under-dispersed. Replicates are pooled. For the short subcritical chains, which die inside six weeks, k and the zero share use every node, because the leaves are exactly the cases with no offspring and dropping the last 14 days would delete them. Venue concentration is the mean across replicates of the share of infections sitting in the busiest 10% of venues that had at least one infection. Stacking venue counts across replicates would treat the same popular place as one bigger place and overstate that share.

On the uncontrolled baseline replicate, the moment estimate of k is 4.45 and a negative-binomial likelihood fit is 4.31. The tables are moment estimates.

## Experiment 1: variance partition

The proposal asks how much of model k comes from an agent multiplier, from schedule heterogeneity, and from venue amplification. The claim to test is that schedules plus venue amplification can produce k below 1 when occupancy is heavy-tailed, while a multiplier alone can hit a target k and still fail to concentrate infections in a minority of venues.

Each cell is eight weeks, no closures, quarantine 0, `-r`, infection coefficient 0.00041, four replicates, seeds 1 to 4. Shedding shape is off, 1, 0.3, or 0.1. Schedules are either the real assignments or shuffled inside each location type. The venue profile is off, or covid at strength 1. One extra cell draws events with daily probability 0.02 and factor 15. A sweep with two replicates then raises the venue strength and the event size.

R is mean censored offspring. p80 is the smallest share of those infectors who account for 80% of offspring. Home is the share of infections acquired at home. v10 is the mean, across replicates, of the infection share in the top 10% of infected venues. Infections and peak prevalence are means across replicates.

| cell | R | k | zero | top 20% | p80 | home | v10 | infections | peak prevalence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shed off, real schedules, venues off | 1.360 | 4.56 | 0.307 | 0.506 | 0.421 | 0.462 | 0.483 | 36,753 | 13,156 |
| shed off, shuffled, venues off | 1.357 | 4.39 | 0.308 | 0.508 | 0.420 | 0.457 | 0.491 | 35,862 | 12,838 |
| shed off, real, covid strength 1 | 1.159 | 2.68 | 0.379 | 0.543 | 0.389 | 0.351 | 0.661 | 144,978 | 50,506 |
| shed off, shuffled, covid strength 1 | 1.158 | 2.61 | 0.381 | 0.546 | 0.387 | 0.355 | 0.658 | 144,417 | 50,224 |
| shape 1, real, venues off | 1.307 | 0.90 | 0.438 | 0.634 | 0.308 | 0.421 | 0.489 | 23,980 | 8,070 |
| shape 1, shuffled, venues off | 1.310 | 0.88 | 0.440 | 0.636 | 0.307 | 0.413 | 0.502 | 25,140 | 8,379 |
| shape 1, real, covid strength 1 | 1.189 | 0.71 | 0.480 | 0.664 | 0.282 | 0.314 | 0.686 | 135,981 | 44,950 |
| shape 1, shuffled, covid strength 1 | 1.207 | 0.70 | 0.478 | 0.664 | 0.282 | 0.317 | 0.684 | 135,362 | 45,529 |
| shape 0.3, real, venues off | 1.215 | 0.29 | 0.604 | 0.805 | 0.197 | 0.364 | 0.512 | 12,146 | 3,481 |
| shape 0.3, shuffled, venues off | 1.206 | 0.29 | 0.608 | 0.807 | 0.196 | 0.364 | 0.512 | 11,018 | 3,200 |
| shape 0.3, real, covid strength 1 | 1.278 | 0.26 | 0.606 | 0.816 | 0.190 | 0.260 | 0.719 | 118,415 | 38,296 |
| shape 0.3, shuffled, covid strength 1 | 1.298 | 0.27 | 0.605 | 0.815 | 0.190 | 0.265 | 0.714 | 115,660 | 37,956 |
| shape 0.1, real, venues off | 1.063 | 0.11 | 0.764 | 0.966 | 0.099 | 0.317 | 0.537 | 5,358 | 1,095 |
| shape 0.1, shuffled, venues off | 1.084 | 0.11 | 0.767 | 0.970 | 0.095 | 0.319 | 0.542 | 5,586 | 1,325 |
| shape 0.1, real, covid strength 1 | 1.347 | 0.094 | 0.755 | 0.967 | 0.095 | 0.196 | 0.740 | 83,621 | 27,731 |
| shape 0.1, shuffled, covid strength 1 | 1.385 | 0.094 | 0.754 | 0.967 | 0.095 | 0.200 | 0.737 | 78,433 | 27,534 |
| events, probability 0.02, factor 15 | 1.385 | 4.12 | 0.304 | 0.509 | 0.419 | 0.445 | 0.519 | 51,634 | 18,723 |

The sweep keeps shedding off and the real schedules:

| cell | R | k | zero | top 20% | home | v10 | infections | peak prevalence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| events, probability 0.05, factor 40 | 1.263 | 1.71 | 0.372 | 0.563 | 0.374 | 0.639 | 130,276 | 42,790 |
| events, probability 0.10, factor 30 | 1.115 | 1.44 | 0.424 | 0.590 | 0.343 | 0.672 | 150,150 | 52,862 |
| covid strength 2 | 1.013 | 1.42 | 0.460 | 0.609 | 0.284 | 0.726 | 165,930 | 75,146 |
| covid strength 4 | 0.998 | 0.85 | 0.508 | 0.667 | 0.214 | 0.791 | 174,878 | 103,970 |
| covid strength 8 | 0.996 | 0.54 | 0.557 | 0.731 | 0.154 | 0.844 | 177,804 | 129,468 |

What the grid shows:

- Real schedules without a multiplier or a venue profile give k = 4.56. The top fifth of infectors cause half of offspring, and 42% of infectors are needed to reach 80%. That is a mild tail. Shuffling at matched occupancy changes k from 4.56 to 4.39 and v10 from 0.48 to 0.49. The mild tail is the occupancy distribution, not who shares a household.
- The covid profile at strength 1 lowers k to 2.6 or 2.7 and raises v10 from about 0.48 to about 0.66. It stays above 1. The same profile on this infection coefficient also infects on the order of 145,000 people in eight weeks, against about 37,000 in the flat run.
- A multiplier alone tracks its shape: 1 gives k of 0.90, 0.3 gives 0.29, 0.1 gives 0.11. At shape 0.1, p80 is 0.099, which is the usual "10% cause 80%" check. v10 stays near the flat baseline (0.54 versus 0.48). The multiplier does not create setting concentration.
- Shape 0.1 with venues off leaves about 5,400 infections and a prevalence peak near 1,100. The gamma mean is 1, and the mass of a shape-0.1 draw sits below 1, so most infectious people are less infectious than the calibrated average. The epidemic shrinks. Putting the same shape next to the covid profile restores a large epidemic (about 80,000 infections) while k stays near 0.09, and v10 rises to 0.74. The venue profile, not the multiplier, is what concentrates places.
- Events at probability 0.02 and factor 15 leave k at 4.12. Harsher events reach k of 1.71 and 1.44. Occupancy dilution in the hazard is why the gentle event is almost invisible.
- Strength 4 is the first cell with k below 1 from venues alone (k = 0.85, v10 = 0.79, household share 0.21). Strength 8 reaches k = 0.54 and v10 = 0.84. The floor in this sweep is still several times the published 0.1, and final size is the whole population.

The proposal's claim holds for the multiplier: it can hit a target k, and setting concentration stays near the baseline until venues are amplified. The claim that schedules plus modest venue amplification produce k below 1 does not hold on this kernel. k below 1 from places alone required strength 4 or more, and that strength saturates the city.

## The fitted first two waves

These jobs are 37 weeks, the fitted closure file, quarantine policy 3, `-r`, seeds 101 and 102, infection coefficient 0.00041. Four corners: the fitted model; shape 0.1 with shuffled schedules; the covid profile at strength 1 on the real schedules; and shape 0.3 plus that profile. Nothing here refits k.

Hospital and prevalence peaks are split at day 140, so the November wave and the March wave are separate. Dates are day 0 plus the peak day, with day 0 equal to 23 September 2020.

| corner | R | k | zero | top 20% | p80 | home | v10 | infections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.995 | 2.61 | 0.436 | 0.569 | 0.365 | 0.564 | 0.434 | 50,417 |
| biology, shape 0.1, shuffled | 0.930 | 0.094 | 0.767 | 0.965 | 0.101 | 0.371 | 0.542 | 8,190 |
| setting, covid strength 1 | 0.997 | 1.95 | 0.448 | 0.586 | 0.353 | 0.475 | 0.549 | 108,939 |
| stack, shape 0.3 plus covid | 0.998 | 0.254 | 0.642 | 0.838 | 0.181 | 0.361 | 0.611 | 70,688 |

| corner | wave-1 hospital | wave-2 hospital | wave-1 prevalence | wave-2 prevalence |
|---|---|---|---|---|
| baseline, rep 1 | 146 on 28 Nov 2020 | 203 on 25 Mar 2021 | 3,000 on 14 Nov 2020 | 3,465 on 20 Mar 2021 |
| baseline, rep 2 | 148 on 13 Dec 2020 | 247 on 3 Apr 2021 | 2,260 on 19 Nov 2020 | 4,327 on 27 Mar 2021 |
| biology, rep 1 | 22 on 12 Oct 2020 | 107 on 18 Mar 2021 | 412 on 1 Oct 2020 | 1,776 on 11 Mar 2021 |
| biology, rep 2 | 30 on 19 Oct 2020 | 26 on 22 Feb 2021 | 478 on 2 Oct 2020 | 464 on 17 Apr 2021 |
| setting, rep 1 | 1,423 on 21 Nov 2020 | 50 on 26 Mar 2021 | 29,401 on 11 Nov 2020 | 726 on 13 Mar 2021 |
| setting, rep 2 | 1,490 on 18 Nov 2020 | 45 on 22 Mar 2021 | 29,032 on 11 Nov 2020 | 1,010 on 13 Mar 2021 |
| stack, rep 1 | 671 on 20 Nov 2020 | 71 on 18 Mar 2021 | 13,577 on 12 Nov 2020 | 1,269 on 13 Mar 2021 |
| stack, rep 2 | 765 on 21 Nov 2020 | 70 on 25 Mar 2021 | 14,787 on 12 Nov 2020 | 1,104 on 12 Mar 2021 |

The baseline is the fitted model. Wave-1 hospital peaks of 146 and 148 sit on the scaled official peak of about 144. Wave-2 peaks of 203 and 247 bracket the scaled peak of about 225. Prevalence peaks are the same order as the scaled reconstruction (about 2,980 and 3,960). Emergent k on this fit is 2.61, and 37% of infectors account for 80% of offspring.

Shape 0.1 produces k = 0.094 and collapses the first hospital wave to 22 and 30. The covid profile at strength 1, still on the fitted coefficient, produces k = 1.95 and a first hospital wave of about 1,450, roughly ten times the fit. The second wave is then small because the first wave has already burned through susceptibles. The stack reaches k = 0.25 with a first hospital peak of 671 and 765.

Heterogeneity stacked on a coefficient that was tuned without it does not preserve the hospital fit. A later calibration could trade a lower coefficient against a heavier tail. This study did not do that trade.

## Experiment 3: what detection does to k

The proposal asks how far k estimated from detected clusters differs from the true simulated k. The trees are the ones already run. The observation models, applied to replicate 1, are:

- Symptom-forward keeps diagnosed index cases and counts only diagnosed children.
- Mixed backward keeps an index case who is diagnosed or who has a diagnosed child. A diagnosed index case contributes every child. An undiagnosed index case contributes only diagnosed children.
- Pure backward keeps infectors with at least one diagnosed child and scores them by their true offspring. Cases with no diagnosed child, including every zero-offspring case, are dropped.
- Cluster investigates a venue with at least three infections and at least one diagnosis, and counts offspring only inside the observed cases.

| trace | model | n | R | k | zero | p80 |
|---|---|---:|---:|---:|---:|---:|
| flat baseline | true | 18,978 | 1.352 | 4.45 | 0.311 | 0.418 |
| flat baseline | symptom-forward | 6,977 | 0.532 | 3.64 | 0.605 | 0.288 |
| flat baseline | mixed backward | 11,031 | 1.354 | infinite | 0.184 | 0.546 |
| flat baseline | pure backward | 6,808 | 2.327 | infinite | 0 | 0.614 |
| flat baseline | cluster | 13,095 | 1.051 | 2.17 | 0.429 | 0.360 |
| shape 0.1, venues off | true | 7,187 | 1.113 | 0.115 | 0.757 | 0.100 |
| shape 0.1, venues off | symptom-forward | 2,770 | 0.469 | 0.096 | 0.830 | 0.083 |
| shape 0.1, venues off | mixed backward | 3,455 | 1.382 | 0.212 | 0.599 | 0.179 |
| shape 0.1, venues off | pure backward | 1,155 | 6.065 | 1.00 | 0 | 0.445 |
| shape 0.1, venues off | cluster | 5,427 | 0.947 | 0.092 | 0.808 | 0.078 |
| covid strength 1 | true | 117,259 | 1.135 | 2.61 | 0.386 | 0.387 |
| covid strength 1 | symptom-forward | 44,159 | 0.435 | 1.98 | 0.672 | 0.240 |
| covid strength 1 | mixed backward | 67,158 | 1.183 | 233 | 0.255 | 0.509 |
| covid strength 1 | pure backward | 37,462 | 2.222 | infinite | 0 | 0.599 |
| covid strength 1 | cluster | 88,139 | 0.918 | 1.61 | 0.482 | 0.335 |
| shape 0.1 plus covid | true | 57,809 | 1.284 | 0.088 | 0.758 | 0.094 |
| shape 0.1 plus covid | symptom-forward | 22,359 | 0.488 | 0.085 | 0.834 | 0.080 |
| shape 0.1 plus covid | mixed backward | 28,075 | 1.571 | 0.160 | 0.604 | 0.164 |
| shape 0.1 plus covid | pure backward | 9,435 | 7.035 | 0.67 | 0 | 0.431 |
| shape 0.1 plus covid | cluster | 49,882 | 1.167 | 0.074 | 0.789 | 0.079 |
| fitted wave, baseline | true | 48,202 | 0.995 | 2.68 | 0.435 | 0.366 |
| fitted wave, baseline | symptom-forward | 15,514 | 0.336 | 1.39 | 0.737 | 0.196 |
| fitted wave, baseline | mixed backward | 23,852 | 1.152 | infinite | 0.248 | 0.521 |
| fitted wave, baseline | pure backward | 12,419 | 2.119 | infinite | 0 | 0.603 |
| fitted wave, baseline | cluster | 31,277 | 0.819 | 1.61 | 0.523 | 0.313 |

Pure backward multiplies mean offspring by 1.7 in the flat baseline (2.33 against 1.35), by 2.1 in the fitted wave (2.12 against 1.00), by 5.4 when the true k is 0.115 (6.07 against 1.11), and by 5.5 when the true k is 0.088 (7.04 against 1.28). That is the direction in the tracing literature: the people found by working backwards have more offspring than a random case. The same filter removes the zeros, so the moment k gets larger. The detected tail looks less overdispersed, not more. A k of several hundred, or an infinite k, on the mixed-backward rows means the retained offspring counts are close to Poisson.

Cluster sampling lowers k: 4.45 to 2.17, 2.61 to 1.61, 2.68 to 1.61, 0.115 to 0.092, 0.088 to 0.074. Symptom-forward lowers the mean, because undiagnosed children are dropped, and moves k by a smaller amount. In the flat baseline, 2,820 venues meet the cluster rule and they hold 62% of infections. In the fitted wave, diagnosed index cases are 32% of censored index cases.

Offspring split by where the child was infected:

| trace | home R | home k | community R | community k |
|---|---:|---:|---:|---:|
| flat baseline | 0.594 | 6.16 | 0.757 | 2.03 |
| shape 0.1, venues off | 0.281 | 0.230 | 0.832 | 0.083 |
| covid strength 1 | 0.393 | 4.23 | 0.743 | 1.57 |
| shape 0.1 plus covid | 0.238 | 0.220 | 1.047 | 0.068 |
| fitted wave, baseline | 0.556 | 3.40 | 0.439 | 0.838 |

Household offspring are less dispersed than community offspring in every one of these traces. That is the same direction as the Hong Kong contrast of household k near 1.1 and retail k near 0.05. The levels are not. Community k in the fitted wave is 0.84. Nothing here produces a retail-like 0.05 without a shedding shape that was put in by hand.

Per-venue cluster size, counting only places with at least one infection, for types that have hundreds of places. A dispersion on the eight restaurants or the twenty large-evening locations is not a useful k: those types are a handful of sites with a high mean, because many visits share them.

| trace | type | places | mean infections | k |
|---|---|---:|---:|---:|
| flat baseline | home | 9,393 | 1.99 | 0.35 |
| flat baseline | short shop | 676 | 4.88 | 1.04 |
| flat baseline | long visit | 890 | 4.12 | 0.92 |
| flat baseline | classroom | 524 | 3.03 | 6.90 |
| covid strength 1 | home | 29,260 | 1.76 | 5.29 |
| covid strength 1 | short shop | 864 | 8.46 | 0.52 |
| covid strength 1 | long visit | 1,253 | 17.15 | 0.40 |
| covid strength 1 | classroom | 739 | 6.84 | 11.7 |
| fitted wave | home | 12,085 | 2.27 | 0.89 |
| fitted wave | short shop | 686 | 6.55 | 0.64 |
| fitted wave | long visit | 912 | 4.85 | 0.70 |
| fitted wave | classroom | 354 | 2.67 | 5.99 |

Homes in the uncontrolled flat run are overdispersed (k = 0.35) even though individual offspring k is 4.45. Amplifying community venues pulls the heavy places out of the household distribution: home k rises to 5.29, and long-visit places go to a mean of 17 with k = 0.40. Setting concentration and individual-offspring k are different statistics. A model can have one without the other, which is what the shape-0.1, venues-off cell already showed.

The audit's conclusion: ascertainment bias is real, and it is too small to explain a simulated k of 2.6 to 4.5 against a published k near 0.1. Published k should still be treated as a range after a detection model. It should not be treated as a number this kernel is missing only because of how cases were found.

## Experiment 2: same window k, different early epidemics

The proposal asks whether two parameterisations with the same marginal k and R rank interventions differently. The claim to test is that a biology-dominated arm favours isolation and responds weakly to venue controls, and a setting-dominated arm does the opposite.

Set I is shedding shape 1, shuffled schedules, venues off, and infection coefficient `0.00041 * (0.998 / 1.307)`. The ratio matches the eight-week mean offspring of the shape-1 flat cell (1.307) to the eight-week mean offspring of venue strength 4 (0.998). Set II is shedding off, real schedules, covid profile at strength 4, and infection coefficient 0.00041.

Both arms run eight weeks with `-r`, no closures, three replicates, and seeds 501 to 503. The levers are present from day 0:

- none
- `communityScale` 0.5, a uniform cut on community-location infectiousness (public, home, hospital, and nursing are left alone)
- the infection coefficient multiplied by 0.75
- quarantine policy 1, the diagnosed agent
- quarantine policy 2, the agent and the household

`communityScale` is not a cap that binds only when a room is full. The kernel has no hard occupancy limit. The contact cut is a lower transmission coefficient from the first step, not a behaviour change partway through the wave.

Over the whole censored window the arms look matched. Set I with no intervention has R = 1.072 and k = 0.796. Set II has R = 0.998 and k = 0.858.

The first week is not matched. Index cases infected on days 0 to 7, with children counted only if they were infected within 14 days:

| arm | replicate offspring n, R, k |
|---|---|
| set I | 903, 1.008, 0.65; 708, 0.853, 0.88; 893, 0.974, 1.05 |
| set II | 6,567, 3.344, 3.14; 4,870, 3.458, 3.30; 6,938, 3.305, 3.34 |

Pooled first-week R is 0.95 in set I and 3.36 in set II. Pooled first-week k is 0.81 and 3.25. On days 0 to 3 the pooled R values are 0.89 and 3.69. Prevalence on day 14 is 582 in set I and 40,721 in set II.

The eight-week average hid that gap because set II grows fast and then runs out of susceptibles, which pulls mean offspring back toward 1, while set I never takes off. The protocol asked for a match on early generations. These two arms share a depleted-window k. They do not share an early reproduction number. The comparison is still the one that decides whether the split matters: the mechanisms diverge by an order of magnitude in early size, and the question is whether the levers swap rank across that divergence.

| arm | infections | peak prevalence | peak hospital | quarantine person-days | day-14 prevalence | early R |
|---|---:|---:|---:|---:|---:|---:|
| I, none | 5,549 | 1,151 | 64 | 0 | 582 | 0.95 |
| I, community half | 1,733 | 435 | 31 | 0 | 336 | 0.71 |
| I, contacts 0.75 | 1,945 | 440 | 31 | 0 | 365 | 0.75 |
| I, isolate | 4,190 | 786 | 49 | 13,185 | 504 | 0.87 |
| I, household | 2,834 | 508 | 38 | 36,571 | 447 | 0.80 |
| II, none | 174,898 | 104,048 | 4,682 | 0 | 40,721 | 3.36 |
| II, community half | 159,895 | 65,908 | 3,401 | 0 | 9,176 | 2.50 |
| II, contacts 0.75 | 169,445 | 85,115 | 4,067 | 0 | 15,696 | 2.92 |
| II, isolate | 175,264 | 103,596 | 4,691 | 687,523 | 36,590 | 3.28 |
| II, household | 170,310 | 95,523 | 4,354 | 1,726,764 | 32,862 | 3.18 |

Against its own control, set I loses 69% of infections under the community cut, 65% under the contact cut, 49% under household quarantine, and 24% under isolation. Peak prevalence falls by 62%, 62%, 56%, and 32%. Infections averted per quarantine person-day are 0.103 for isolation and 0.074 for household quarantine.

Set II loses 9% of infections under the community cut and the peak falls 37%. The contact cut removes 3% of infections and 18% of the peak. Household quarantine removes 3% of infections and 8% of the peak. Isolation does not reduce final size (175,264 against 174,898) or the peak. Both tracing policies spend hundreds of thousands to millions of person-days once almost everyone is reachable.

Community hazard reduction is the strongest lever in both arms. Isolation is the weakest in both. High shedders still transmit in community venues, and diagnosis arrives after much of that transmission. After the venue-driven arm has saturated the population, final size barely moves, and the peak still moves most under the community cut.

The proposed ranking is not supported. The split that changes the decision is whether the epidemic saturates, not whether isolation outranks a venue control.

## Experiment 4: subcritical chains

The proposal asks whether, at the same mean R below 1, caregiving and health-care heterogeneity raises the chance of a rare large cluster and the share of cases with no offspring. The reference patterns are Nipah-like R near 0.3, with a literature figure of about 92% of cases transmitting to nobody, and H5N1-like R at or below about 0.25. H5N1 has no estimated human-to-human k. These jobs keep the Szeged schedules and the coronavirus natural history, and they relabel exposure with the caregiving profile. They are illustrative scenarios. They are not a model of outbreaks in Bangladesh or India, and they are not an influenza forecast.

A four-week pilot at the fitted coefficient 0.00041, two replicates, and no `-r`, gave pooled generation-1 R of 1.14 (flat, schedules shuffled, 7 seeds), 1.29 (caregiving, no multiplier), and 1.86 (caregiving plus shape 1). Chains reached from the tens into 160 cases, because reproduction was above 1. The coefficient had to come down before the hypothesis could be tested.

A first pass still overshot. Flat, target 0.30, 16 replicates and 82 seeds: generation-1 R 0.43, largest chain 10, zero share 0.64, and k = 6.9, close to Poisson. Flat, target 0.20: generation-1 R 0.32, largest chain 5, zero share 0.73, k infinite. Caregiving, no multiplier, target 0.30: generation-1 R 0.63, largest chain 26, 3.7% of chains above 10, zero share 0.55, k = 2.95. The caregiving target-0.20 cell of that pass was stopped after one replicate that produced no secondary infection, and that cell is unused.

The reported grid is the second pass. Each first-pass coefficient was multiplied by target divided by achieved generation-1 R. The shape-1 arms reused the no-multiplier overshoot factor, which was the wrong correction for them. Twelve replicates, six weeks, no `-r`, seeds 801 to 812. The isolation arm uses the caregiving, no-multiplier, target-0.30 coefficient, quarantine policy 1, and seeds 901 to 912. There are 52 introduction chains in the main arms and 55 in the isolation arm. k and the zero share are computed on every node in the finished trees.

| arm | generation-1 R | largest chain | mean chain | chains above 10 | chains above 50 | zero share | k |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat, shuffled, target 0.30 | 0.327 | 5 | 1.50 | 0 | 0 | 0.718 | infinite |
| flat, shuffled, target 0.20 | 0.231 | 5 | 1.33 | 0 | 0 | 0.768 | infinite |
| caregiving, shape 0, target 0.30 | 0.404 | 24 | 1.98 | 1 of 52 | 0 | 0.650 | 0.64 |
| caregiving, shape 0, target 0.20 | 0.250 | 13 | 1.52 | 1 of 52 | 0 | 0.747 | 0.70 |
| caregiving, shape 1, target 0.30 | 0.135 | 3 | 1.15 | 0 | 0 | 0.883 | 0.98 |
| caregiving, shape 1, target 0.20 | 0.096 | 3 | 1.12 | 0 | 0 | 0.914 | 0.41 |
| caregiving, shape 0, target 0.30, isolate | 0.545 | 15 | 1.84 | 1 of 55 | 0 | 0.683 | 0.75 |

The clean pair is generation-1 R of 0.23 in the flat arm and 0.25 in the caregiving arm, both aimed at 0.20. Caregiving lengthens the extreme chain from 5 to 13 and moves k from infinite to 0.70. The zero share is 0.747 against 0.768. Heterogeneity does not raise the fraction of cases with no offspring. The target-0.30 pair is only partly matched (0.40 against 0.33) and shows the same pattern: largest chain 24 against 5, k = 0.64 against infinite, zero share 0.65 against 0.72.

No calibrated chain reached 50. With about 52 chains, a probability of a few percent would often contribute one event or none. The absence of a size-50 chain is a thin-sample statement.

The shape-1 arms landed at generation-1 R of 0.14 and 0.10. They are not a matched comparison, and they are not evidence about a modest multiplier at R = 0.3. Symptom isolation spent about 39 quarantine person-days across six weeks, left a largest chain of 15, and realised generation-1 R of 0.55 against 0.40 in the uncontrolled caregiving arm with the same coefficient. Diagnosis barely engages when outbreaks stay this small. The arm does not show that isolating index cases shortens chains.

A Poisson draw with mean equal to the flat arm's all-node offspring (0.33 at the 0.30 target) has a zero probability of about 0.72, which is the 0.718 in the table. The literature figure of about 92% zeros at Nipah-like R is a heavier tail than these flat chains, and the caregiving profile moved the zero share down, not up, because the extra variance was paired with a slightly higher mean. The 92% figure is not reproduced.

A few chains were still infecting late in week 6. The last infection day in a replicate reached as high as the mid-30s, out of a six-week horizon. Chain lengths are observed lengths at six weeks.

The first pass, at a higher reproduction number, points the same way and is not the primary result: largest chain 10 at achieved generation-1 R 0.43 in the flat arm, and largest chain 26 at achieved R 0.63 in the caregiving arm.

## What PanSim should do with this

Keep the location schedules as the base layer. They are already the contact structure of the fitted model. The shuffle shows that the gap between this kernel and a Poisson offspring distribution, k of 4.5 rather than infinity in an uncontrolled run, is the occupancy distribution.

Do not place a gamma of shape 0.1 or 0.3 on the fitted infection coefficient without a new calibration. Most draws are below 1, and the hospital curve moves.

Do not expect the present frequency-dependent kernel, plus the modest venue table, to land on a published k near 0.1. Strength 4 to 8 can pass below 1 and does concentrate venues, and it saturates the population and leaves the hospital fit. Explicit events have to be harsh before they matter, because the hazard divides by occupancy.

Pass simulated trees through a detection model before comparing them with a published k, and treat that published k as a range. In these runs the detection model does not close the gap.

Report the split between shedding, contacts, and venues as an assumption. Matching one k, especially a k averaged over a wave that has already depleted susceptibles, does not identify the split and does not match early growth.

If a shedding multiplier is used, draw it for the infection episode. These runs give no support for storing it as a lifelong class. That is the same conclusion as the literature brief: overdispersed transmission is real for SARS-CoV-2, and a distinct persistent superspreader class is not what the evidence shows.

## Caveats

The literature packet was read from abstracts. This study does not extend that reading.

The venue numbers are assumed profiles. They are not attack rates fitted to Hungarian settings.

The infector in the trace is one draw from the people present, weighted by infectiousness. The weighting matches the kernel. It is still an imputation, because the kernel is not pairwise.

Moment estimates are unstable when the variance sits on the mean. Infinite k means the sample is not overdispersed.

Experiment 2 applies a flat community multiplier and a flat coefficient cut from day 0. It does not test a cap that binds only in crowded rooms, and it does not test a lever switched on after the epidemic is visible.

Experiment 4 has about 52 chains per arm. The far tail is not resolved. The shape-1 arms and the isolation arm missed the reproduction number they were aimed at. Scenarios are stylised.

## How to rerun

Raw traces and simulator logs under `superspreader/study/results/` are gitignored. The tables in this note are the moment summaries in `superspreader/study/summaries/`.

```
python3 superspreader/study/run_study.py exp1
python3 superspreader/study/run_study.py sweep
python3 superspreader/study/run_study.py waves
python3 superspreader/study/run_study.py exp4pilot
python3 superspreader/study/run_followup.py cal
python3 superspreader/study/summarize_tables.py
```

`run_followup.py cal` is the calibrated chain grid and experiment 2. The runners skip a job that already has a stdout file and a non-empty trace. One GPU run at a time. The binary is `build_gpu/panSim`.
