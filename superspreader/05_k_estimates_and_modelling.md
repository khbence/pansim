# ACT-049 Secondary sections: k estimates and modelling approaches

Depends on the conclusions in 01_literature_review.md. Citations [R#] refer to the reading list in that file. Only values read from verified abstracts are shown as literature values; anything computed here is labelled.

## A. Definitions in brief

See section 2 of the review. In short: nu is a person's expected number of secondary cases in a context, R is its mean, k is the shape of the gamma distribution of nu (equivalently the negative-binomial dispersion), the 20/80 pattern is a share statistic that depends on k, and superspreading events are cluster-level outcomes that may or may not reflect superspreading individuals.

## B. Published k and related estimates

Values are as reported in the abstract of each source; "not in abstract" means the full paper may contain more and should be checked before any number is used as a calibration target.

| Pathogen | Quantity reported | Value (uncertainty) | Source | Setting and period | Method |
|---|---|---|---|---|---|
| SARS-CoV-2 | k | about 0.1 (95% CrI 0.05-0.2 at R0 = 2.5); joint range R0 1.4-12, k 0.04-0.2 | [R3] | Outbreak sizes outside China, early 2020 | Negative-binomial branching process fitted to aggregated counts; upper R0 bound not informed by data |
| SARS-CoV-2 | Share of cases seeding 80% of local transmission | 19% (95% CI 15-24%); k not in abstract | [R4] | Hong Kong, 23 Jan-28 Apr 2020, 1,038 cases | Cluster analysis of contact-tracing data |
| SARS-CoV-2 | Share of primary infections giving 80% of secondary | 15%; k and interval not in abstract | [R5] | Hunan, China, early 2020 | Contact-tracing data with transmission model |
| SARS-CoV-2 | k by setting | 0.05 (retail) to 1.1 (households); eight settings in total; intervals not in abstract | [R6] | Hong Kong 2020-2021, 8,647 cases | Setting-stratified cluster analysis |
| SARS-CoV-2 | Mean k across studies | range 0.04 to 2.97; 93% of 28 studies below 1 | [R7] | Systematic review, studies from 2020 onward | Pooled reported estimates |
| SARS-CoV-2 | Sensitivity to offspring distribution | Poisson-lognormal fitted two of three datasets better than negative binomial | [R8] | Hong Kong, India, Rwanda | Simulation and reanalysis |
| SARS-CoV (2003) and others | k | reported in [R2] for eight diseases; values not verified in this run | [R2] | Contact-tracing datasets | Check the paper before use |
| Nipah | Reproduction number | R = 0.33 (95% CI 0.19-0.59); 1.1 (0.4-3.2) for age 45+ with breathlessness | [R21] | Bangladesh 2001-2014, 248 cases | Regression on outbreak-investigation data |
| Nipah | Proportion of cases that transmitted | 9 of 122 (7%) | [R18] | Bangladesh 2001-2007 | Descriptive |
| Nipah | Proportion of cases with no secondary cases | 92.0% (95% CrI 87.0-95.4); k not in abstract | [R22] | 23 studies, Bangladesh and India | Bayesian superspreading-risk model |
| Nipah | Summary of R estimates | four of five central estimates below 1; no dispersion estimate reported in the abstract | [R23] | Systematic review to March 2025 | Review |
| H5N1 | Human-to-human R | 0.1-0.25, upper bounds below 0.4 | [R26] | Indonesian households 2005-2009 | Household final-size model |
| H5N1 | R | below 0.2 | [R28] | Rapid review of 46 articles | Review plus branching-process simulation |
| H5N1 | k | None located | - | - | - |

**Where none exist:** no H5N1 human-to-human k and no verified Nipah k were found in the sources screened. Use scenario values and label them as assumptions.

### Reference table computed here (not from the literature)

If nu is gamma distributed with mean 1 and shape k, the fraction of individuals (ranked by nu) who account for 80% of expected transmission, and the share of transmission from the top 20%, are:

| k | Fraction of individuals accounting for 80% of expected transmission | Share of expected transmission from the top 20% |
|---|---|---|
| 0.05 | 5.3% | 99.5% |
| 0.1 | 9.8% | 95.1% |
| 0.16 | 14.5% | 88.3% |
| 0.3 | 23.1% | 75.7% |
| 0.5 | 31.6% | 65.0% |
| 1 | 43.9% | 52.2% |
| 2 | 54.6% | 42.4% |
| 5 | 64.8% | 33.8% |

These are expectations of nu, not realised offspring (realised counts add Poisson noise). The k = 0.1 row (9.8%) agrees with the "about 10% of cases cause 80%" quoted by [R3], which serves as a sanity check. The 19% (Hong Kong [R4]) and 15% (Hunan [R5]) shares correspond to k of roughly 0.16-0.25 under this approximation, but those studies counted realised offspring so this equivalence is indicative only.

For Nipah, the probability of zero offspring at R = 0.33 is 72% under Poisson, 75% at k = 1, 80% at k = 0.3 and 86% at k = 0.1 (negative-binomial formula (1 + R/k)^(-k)), computed here. The 92-93% non-transmitter proportions [R18, R22] exceed all of these, which is why the Nipah data point to strong heterogeneity, subject to the data caveats in the review.

## C. Modelling approaches

### C1. Negative-binomial offspring and branching processes
- **What:** Draw each case's offspring from a negative binomial with mean R and dispersion k, equivalently a Poisson draw with gamma-distributed mean [R2]. Outbreak extinction probability, final size distribution and early-growth variability follow from the offspring generating function.
- **Use:** Quick scoping of outbreak risk for subcritical or near-critical pathogens, and estimating k from cluster-size data [R3, R8]. For Nipah and H5N1 this is the natural framework because data come as stuttering chains [R22, R26].
- **Limits:** No mechanism, space or time; the negative binomial is an assumption that other distributions can beat [R8]; the estimate depends on how clusters were detected [R9, R10].

### C2. Gamma-distributed individual infectiousness in agent-based models
- **What:** Each agent receives a multiplier drawn from a gamma distribution with mean 1 and shape k that scales its per-contact hazard [R29]. Alternatives are lognormal multipliers or an individual viral-load trajectory with a person-specific peak and timing [R12, R13].
- **Behaviour:** In [R29], with low k, reducing contacts between people who do not regularly meet had far more effect than reducing regular contacts, so conclusions about interventions depend on this choice.
- **Pitfall:** In an agent-based model the realised k comes from the multiplier and from contact heterogeneity and chance. If both are included and the multiplier is tuned to the published k on its own, the model double counts. See the identifiability section in 03.

### C3. Contact-network heterogeneity
- **What:** Heterogeneity arises from who meets whom and for how long: layered networks (close, regular, random) [R29], or mobility networks from agents to points of interest [R30]. In [R30] a small minority of venues account for most infections and occupancy caps outperform uniform mobility cuts.
- **Link to PanSim:** This is the natural home of the ACT-004 schedule and POI-to-agent assignment, provided agents' venue visits, durations and co-presence are stored so that exposure-time at each venue can be computed.
- **Limits:** The tail is only as heavy as the schedule model allows; survey-derived schedules typically truncate rare extremes. Emergent k is an output, not a knob.

### C4. Setting-specific transmission
- **What:** Per-venue transmission intensity combining ventilation, crowding, duration and activity, for example the infectious-quanta framework applied to the Skagit chorale outbreak [R11], setting-stratified k and cluster sizes [R6], and venue occupancy caps [R30].
- **Evidence base:** The most convergent of the four explanations (review section 7). It is also the one that most naturally explains "events" without positing special individuals.
- **Limits:** Parameters (quanta per hour, air-change rates) come from few well-characterised events [R11] and are not directly transferable to Hungarian venues.

## D. How the approaches follow from the review

| Review conclusion | Modelling consequence |
|---|---|
| Overdispersion is supported but its attribution is unresolved | Do not hard-code one mechanism; implement factors separately and switchable |
| Setting and events are the best-supported driver | Setting-specific transmission and POI-level exposure should be present in any SARS-CoV-2 scenario |
| Biological heterogeneity is large but transient and weakly passed through to offspring | Use a modest per-agent multiplier or a time-varying viral-load profile, not a lifelong supershedder class |
| Nipah and H5N1 evidence is thin | Treat k as a scenario parameter; focus on health-care and caregiving settings (Nipah) and keep H5N1 as a clearly labelled hypothetical |
