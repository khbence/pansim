# ACT-049 Brief: do superspreaders exist, and what should PanSim do about it?

For István Reguly. Evidence cut-off 9 October 2026. Summary of the full review (01_literature_review.md); citations [R#] refer to its reading list. Public literature only.

## 1. Answer to the central question

**Overdispersed transmission is real for SARS-CoV-2; a distinct class of "superspreaders" is not demonstrated.** Reported dispersion k ranges from 0.04 to 2.97 across 28 studies, with 93% below 1 [R7], so a minority of cases cause most secondary infections. But that is an outcome. Three explanations compete, and the evidence ranks them unevenly:

- **Setting and event effects: supported.** Clusters concentrate in particular, mostly indoor settings [R10]; k differs by setting (about 0.05 in retail versus about 1.1 in households in Hong Kong [R6]); a modelled choir event is consistent with aerosol build-up during a long, poorly ventilated rehearsal [R11]; a POI-based model finds a few venues responsible for most infections [R30].
- **Behavioural (contact and mobility): mixed.** Exposure duration and closeness raise transmission risk [R5] and social settings generate more secondary cases than households [R4]. But no verified study shows a persistent individual contact trait driving offspring numbers, and behaviour is confounded with setting in tracing data.
- **Biological (shedding, viral load, aerosol emission): mixed.** Emission varies by about three orders of magnitude between people [R16] and infectious virus shedding differs between individuals [R13, R15]. Yet in a linked cohort the index viral load moved secondary attack rate only from 12% to 24% [R14], infectious windows are short [R17], and no study links measured emission to realised offspring.

The most defensible reading is a multiplicative model: shedding x contact intensity x setting amplification, each heavy-tailed, none sufficient alone [R12]. That reading is our interpretation across sources, not a published result. Superspreaders as a persistent, distinct class: **weak**.

## 2. Graded conclusions by pathogen

| | SARS-CoV-2 | Nipah | H5N1 (human-to-human) |
|---|---|---|---|
| Overdispersion observed | Supported | Mixed | Weak (not assessable) |
| Behavioural | Mixed | Mixed (caregiving exposure, not mobility) | Weak |
| Biological | Mixed | Weak | Weak |
| Setting and events | Supported | Mixed | Weak |

**Nipah** shows few transmitters (7% of 122 cases [R18]; R = 0.33 in 248 cases [R21]; 92% non-transmitters in one pooled model [R22]), linked to long caregiver exposure, body fluids, health-care and burial settings [R19, R20, R21]. **Evidence is thin**: tiny counts, subcritical, outbreak-driven ascertainment, and a systematic review notes that transmission estimates are scarce [R23]. **H5N1** has essentially no human-to-human transmission to disperse (R about 0.1-0.25 [R26, R28]; no household transmission among 97 contacts in 2024 US cases [R27]); no dispersion estimate exists, and any superspreading scenario is hypothetical.

## 3. Methodological cautions

Detected clusters over-represent large events [R10]; backward tracing finds primary cases with 3-10 times more offspring than random cases [R9]; k depends on the assumed offspring distribution [R8]; household k is capped by household size; early estimates used aggregated data [R3]. Published k should therefore be treated as a range, not a constant.

## 4. Secondary sections in brief

- **Definitions.** nu is an individual's expected offspring; R its mean; k the gamma shape of nu (negative-binomial dispersion); 20/80 is a share statistic that depends on k (k = 0.1 gives about 10% of individuals accounting for 80%, computed here); superspreading events are cluster outcomes, not necessarily the work of superspreading individuals [R1, R2].
- **k estimates.** SARS-CoV-2 about 0.1 early on [R3], 0.05-1.1 by setting [R6], 0.04-2.97 across studies [R7]. Nipah: R = 0.33 [R21] and 92% zero offspring [R22]; no verified k. H5N1: none. Full table in 05_k_estimates_and_modelling.md.
- **Modelling approaches.** Negative-binomial branching process for risk scoping; gamma-distributed individual infectiousness in agent-based models [R29]; contact-network and POI heterogeneity [R30]; setting-specific transmission using quanta, ventilation, duration and occupancy [R11, R6].

## 5. PanSim recommendation


> **Update after phase 2 and 3 (09 and 13).** The configuration that keeps the hospital fit and carries dispersion is a per-infection gamma multiplier of shape 0.3, early k about 0.27 to 0.32. Shape 0.1 and every venue-based corner fail the wave-2 hold-out. The explanation the review grades best supported for SARS-CoV-2, setting and event effects, is the one the present kernel cannot express: expected offspring does not depend on occupancy, so crowding cannot amplify a single infector. Explicit events can (13, section 6), but they are not yet matched on early R.

Implement heterogeneity as a **switchable stack** (full table in 03_pansim_options_table.md): (B) contact heterogeneity through location and schedule assignment from ACT-004 as the base; (A) a modest per-agent multiplier or a viral-load trajectory, not a lifelong supershedder class; and optionally (D) explicit superspreading events. Calibrate in layers: biological variance from independent shedding data, setting parameters to setting-level k, and check that total emergent k lies in the published range after applying a detection model.

**Identifiability:** a single published k fixes total variance, not its split, so tuning a multiplier alone to k would credit biology with what behaviour and venues produce. Behavioural, biological and setting components are only partially separable with public data (independent priors, setting-stratified clusters, time variation, households); the split should be reported as an assumption and probed by ablation.

## 6. Proposed experiments (04_pansim_experiments.md)

1. Variance-partition ablation: which factor generates model k, and how sensitive is k to the split?
2. Same k, different mechanisms: do capacity caps, contact cuts and individual isolation rank differently under biology-dominated versus setting-dominated calibrations?
3. Ascertainment audit: how biased is k estimated from detected clusters relative to true simulated k? Run first.
4. Subcritical chains: Nipah-like (R about 0.3) and H5N1-like (R at most 0.25) scenarios in health-care and household settings, with heterogeneous caregiving exposure.

## 7. What to flag to the reviewer

Two Nipah web-sourced facts (West Bengal January 2026) were not re-fetched and media accounts differed on case numbers; the Nipah superspreading model [R22] and the systematic review [R23] were read from abstracts only; full texts were not read for any item.
