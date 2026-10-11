# ACT-049 Literature review: do superspreaders really exist as a distinct phenomenon?

Primary deliverable of work packet ACT-049 (worker run 9 October 2026, due 23 October 2026). Public literature only. Citations are given as [R1]-[R30] and resolve to the annotated reading list in section 9, every entry of which was checked (see ACT-049_citation_verification.csv).

## 1. Bottom line

1. **Overdispersion is real and robust for SARS-CoV-2; "superspreaders" as a distinct class of people are not demonstrated.** Across many independent datasets the number of secondary cases per case is highly skewed (reported mean k from 0.04 to 2.97 across 28 studies, 93% below 1 [R7]). That is an observation about outcomes. It does not by itself tell us whether the skew comes from who a person is (biology), what they do (behaviour) or where and when they are (setting and event).
2. **The best-supported single driver for SARS-CoV-2 is setting and event structure, with individual biology and behaviour acting as multiplicative modifiers.** Setting effects are supported by several designs [R4, R6, R10, R11, R30]. Biological heterogeneity in shedding and aerosol emission is large [R13, R15, R16] but translates into only a modest gradient in realised transmission in the one cohort with linked contacts (secondary attack rate 12% vs 24% across viral-load strata [R14]). Behavioural heterogeneity matters mostly through exposure duration, closeness and crowding [R5, R12], and in tracing data it cannot be cleanly separated from setting.
3. **The "distinct class" reading is weak.** Nothing in the verified sources shows a bimodal population of persistent supershedders or supermovers. The evidence is more consistent with continuous heterogeneity from several sources, with a tail that produces the events we label superspreading [R2]. Individual labels are also assigned after the fact, from one realised outcome.
4. **For Nipah and H5N1 the evidence is thin.** Nipah shows a consistent pattern of few transmitters (7% of cases transmitted [R18]; 92% could not generate secondary cases in one pooled model [R22]), tied to caregiving exposure, respiratory symptoms and healthcare or burial settings [R19, R20, R21], but the counts are tiny, the process is subcritical and ascertainment is outbreak-driven. For H5N1 there is essentially no human-to-human transmission to disperse: R is estimated below about 0.25 [R26, R28], no dispersion estimate was located, and human cases remain overwhelmingly zoonotic [R27].

Graded conclusions per pathogen and explanation are in section 7.

## 2. Definitions, and where the literature is inconsistent

- **Individual reproduction number (nu).** The expected number of secondary infections caused by a specific infectious individual in a given context. The observed number of secondary cases for that person is one random draw given nu. The population mean of nu is the reproduction number R (R0 in a fully susceptible population) [R2].
- **Dispersion parameter k.** In the standard formulation, nu varies between individuals as a gamma distribution with mean R and shape k, and offspring counts are then negative binomial with variance R(1 + R/k). Small k (below 1) means strong heterogeneity; k tending to infinity recovers the Poisson case [R2]. k is estimated from offspring or cluster-size data, so it describes the observed offspring distribution, mixing everything that varies between cases (biology, behaviour, setting, time, interventions, detection).
- **The 20/80 pattern.** Originally an empirical regularity across disease systems that about 20% of hosts account for at least 80% of transmission potential [R1]. For SARS-CoV-2 it is now used loosely for the share of realised secondary cases (for example 19% of cases seeded 80% of local transmission in Hong Kong [R4]; 15% in Hunan [R5]; about 10% at k = 0.1 [R3]). These are different quantities: the original is a property of a host population's transmission potential, the modern one a property of realised offspring.
- **Superspreading individual vs superspreading event.** An event is a cluster or episode with unusually many infections (a choir rehearsal, a conference, a care home). An individual is a case with unusually many offspring. An event can arise from one high-emitting person, from a favourable setting, or both, and the data usually cannot tell which [R11]. Lloyd-Smith et al. proposed a formal definition of superspreading events in terms of the offspring distribution [R2]; many later papers use "superspreader" for any case in a large cluster, which silently attributes a setting-level outcome to a person.
- **Inconsistencies to keep in view:** (i) k is not a pathogen constant. Hong Kong data give k about 0.05 in retail but about 1.1 in households [R6], and k estimates fall across a wide range between studies [R7]. (ii) nu is a latent trait but is routinely inferred from a single realised offspring count. (iii) The negative binomial is a modelling convenience; other offspring distributions can fit better [R8]. (iv) "Superspreader" mixes three meanings: a persistent personal trait, a transient state (for example peak viral load, early illness) and a lucky draw.

## 3. How the evidence was weighed

Study designs were ranked by how directly they can speak to mechanism and how exposed they are to selection bias. Highest weight: systematic reviews and large linked cohorts with measured exposure. Medium: single-site cohorts, outbreak investigations with a comparison group, setting-stratified cluster analyses. Lower: model-based inference (it depends on assumed structure) and single cluster reports. Each source is tagged with its design in section 9. Evidence was graded:

- **Supported:** several independent studies of different designs converge and known biases are unlikely to reverse the conclusion.
- **Mixed:** findings point the same way but rest on small samples, indirect measures, or are qualified by contrary or confounded evidence.
- **Weak:** few or no direct studies; mostly indirect inference or anecdote.

## 4. SARS-CoV-2

### 4.1 Overdispersion as an observed phenomenon

Different groups, data types and countries report skewed offspring or cluster distributions: negative-binomial fits to outbreak sizes (k about 0.1) [R3], contact-tracing cohorts (19% of cases seeding 80% of local spread in Hong Kong [R4]; 15% in Hunan [R5]), and a systematic review of 28 studies [R7]. The direction is consistent, but magnitude is not: reported mean k spans 0.04 to 2.97 [R7], and part of that spread is method and setting rather than biology.

### 4.2 Behavioural explanation: contact and mobility heterogeneity

For:
- In Hunan tracing data transmission risk rose with duration of exposure and closeness of contact [R5].
- In Hong Kong, social settings produced more secondary cases than households after adjusting for age [R4], and k varied from about 0.05 (retail) to about 1.1 (households) [R6].
- A model linking viral load to offspring distributions concluded that superspreading events need both a high-shedding infector and a large number of exposed contacts [R12].
- In a point-of-interest mobility model a small minority of venues account for most infections, so who goes where and when drives the tail [R30].

Against or qualifying:
- None of the verified sources measures a persistent individual-level behavioural trait (for example a person's contact volume across time) and links it prospectively to realised offspring. Tracing data record settings and exposures, not personal contact propensity. The review's search was not exhaustive, so this is "not found", not "does not exist".
- Behaviour and setting are collinear in observational cluster data. A person with many contacts in a crowded indoor venue contributes both.
- Interventions change contact structure over time, so a single k merges behavioural phases.

### 4.3 Biological explanation: shedding, viral load, aerosol emission

For:
- Exhaled aerosol varied by about three orders of magnitude between 194 healthy subjects, with 18% producing 80% of the aerosol [R16].
- A systematic review and meta-analysis of respiratory viral loads found broad case heterogeneity, with highly infectious cases shedding tens to thousands of virions per minute [R15].
- Daily sampling of 60 people showed substantial person-to-person variation in infectious virus shedding [R13].
- In Catalonia the index case's viral load was associated with transmission (adjusted OR 1.3 per log increase [R14]).

Against or qualifying:
- The viral-load effect on realised transmission is modest in the linked cohort (12% vs 24% secondary attack rate across strata [R14]), far smaller than the three orders of magnitude seen in emission [R16]. Emission heterogeneity has not been shown to carry through to offspring numbers, because [R16] measured healthy subjects.
- Infectious windows are short: live virus was not detected beyond day 9 of illness in any study in a systematic review, and viral load peaks in the first week [R17]. A person is therefore highly infectious only transiently, which is a state, not a lifelong trait.
- Much of the data is cross-sectional or modelled from viral load rather than observed transmission [R12, R13, R15]. A Boulder saliva-screening analysis argued that about 2% of positives carry 90% of circulating virions (Yang et al. 2021, https://doi.org/10.1073/pnas.2104547118); it is not in the reading list because it infers transmission relevance from viral load alone.
- No verified source showed that persistent biological supershedder status is a stable characteristic of an individual across infections or variants.

### 4.4 Setting and event effects

For:
- Clusters are reported across a wide range of mostly indoor settings, with few from schools, many from households and increasing numbers in hospitals and elderly care [R10].
- The Skagit Valley Chorale outbreak (53 of 61 attendees affected) is consistent with aerosol transmission amplified by duration, singing and limited ventilation; a modelled emission of about 970 quanta per hour halves with higher aerosol loss and a shorter event [R11].
- Setting-stratified k and cluster sizes differ markedly across eight settings [R6]; the POI mobility model finds a few venues responsible for most infections and that occupancy caps outperform uniform mobility reduction [R30].
- Phylogenetic analysis of early Boston-area transmission is a further example of events driving local spread (Lemieux et al. 2021, https://doi.org/10.1126/science.abe3261), not in the 30-item list.

Against or qualifying:
- Cluster-by-setting counts are not risks. They are biased by reporting and media attention, as the authors of the review themselves note [R10].
- A setting effect cannot be separated from the biology of the person who happened to attend. The choir event is explained equally well by an unusually high-emitting index and a conducive room [R11].

### 4.5 Confounders and artefacts

1. **Ascertainment and detection bias.** Large clusters are more likely to be noticed and reported [R10]. Contact tracing finds clusters from the other end: backward tracing preferentially identifies primary cases with many offspring, 3-10 times more than a random case under overdispersion [R9]. This makes the "average" detected superspreader look more extreme than the population tail.
2. **Small samples and aggregation.** Early k estimates used aggregated, early-epidemic counts and left the upper bound of R0 uninformed by data [R3].
3. **Estimator and distribution choice.** Variance estimates can be biased under strong heterogeneity, and other distributions (for example Poisson-lognormal) fit some datasets better than the negative binomial [R8].
4. **Time and interventions.** k and R both change as interventions and immunity change; a single pooled k hides this [R6, R7].
5. **Household truncation.** In households the maximum number of secondary cases is bounded by household size, so a higher household k [R6] partly reflects truncation, not an absence of individual variation.

### 4.6 Weighing

The skew in realised outcomes is established. Attribution is the open question. The evidence for a large biological component rests on emission and shedding heterogeneity that is real but has an unquantified, probably weaker, pass-through to offspring numbers [R14, R16]. The evidence for a behavioural component is mostly about exposure duration and closeness in tracing data, indistinguishable from setting. The evidence for setting is the most convergent across designs. The most defensible reading is a multiplicative model in which infectiousness at time t is roughly shedding(t) x contact intensity(t) x setting amplification, with each factor heavy-tailed and none sufficient alone [R12, R2]. This is an interpretation across sources, not a result any single study proved, and its relative weights are unknown.

## 5. Nipah virus

- **Observed pattern:** In Bangladesh 2001-2007 there were 23 introductions and 122 cases; 62 (51%) followed close contact with another case, but only 9 (7%) of cases transmitted [R18]. Pooled over 2001-2014, 82 of 248 cases were person-to-person and R was 0.33 (95% CI 0.19-0.59), with no infections among 1,863 asymptomatic contacts [R21]. A model pooled across 23 studies (33 primary and 152 secondary cases) estimated that 92.0% of cases could not generate secondary cases and a 1.1% chance of an outbreak above 50 cases from 5 seeds [R22].
- **Nuance on the zeros:** If every case had the same R of 0.33, a Poisson process would give about 72% with zero offspring. Observed proportions of non-transmitters (about 93% [R18], 92% [R22]) are higher, which points to heterogeneity, with the caveat that these come from different datasets, definitions and sampling [R18, R22]. The 72% figure is my calculation, not a published value.
- **Behavioural and exposure drivers:** Infection risk rose with exposure over 48 hours (adjusted OR 13) and body-fluid exposure (4.3), and spouses were more often infected than other contacts [R21]. This is caregiver exposure intensity, not mobility of infectious cases. Severe illness probably limits the movement of Nipah cases (inference, not tested in the sources), which makes a "highly mobile superspreader" mechanism less plausible than for SARS-CoV-2.
- **Biological drivers:** Cases with difficulty breathing transmitted more often (12% vs 0%) [R18], and R was higher for older cases with breathlessness [R21]. These are clinical proxies consistent with respiratory shedding. No primary human viral-load or aerosol-emission study was identified among the sources screened. A 2026 narrative review (Gupta et al., Reviews in Medical Virology, https://doi.org/10.1002/rmv.70194, abstract read only, not in the 30) argues that respiratory-sample Ct values below 25 may flag high-risk shedders but also concludes that a standalone viral-load threshold cannot predict a spreading event and that superspreading is multifactorial; it is a narrative synthesis, so it adds little independent evidence. A ferret study in the literature screened (Clayton et al. 2012, https://doi.org/10.3201/eid1812.120875) found higher oral shedding of the Bangladesh strain than the Malaysian strain; that compares strains, not individuals, and is not part of the reading list.
- **Setting and event drivers:** Hospital surfaces were contaminated in Faridpur 2004 [R19]; Faridpur 2010 documented nosocomial and corpse-to-human transmission and an unprotected physician [R20]. The January 2026 West Bengal event involved two health-care workers from one hospital, and WHO reported 196 contacts all asymptomatic and negative as of 27 January 2026 (web sources surfaced in search, including a UN India page relaying WHO, not independently re-fetched; some media reports at the time described more cases and the discrepancy was not resolved here, so the official two-case figure should be rechecked).
- **Systematic review verdict:** parameter estimates are scarce, central R estimates are mostly below 1, and only 8 of 39 models were fitted to data [R23].

## 6. H5N1 (human infections)

- **Human-to-human transmission is rare to absent.** R estimates are about 0.1-0.25 in Indonesian households with upper bounds below 0.4, with strong support for any human transmission only when one large cluster is included [R26]; a rapid review gives R below 0.2 [R28]. Documented probable chains are tiny: a Thai family cluster [R24] and Indonesian family clusters in which the source of infection in two of the three index patients was not determined [R25].
- **US 2024 cases:** 46 cases, mild, mostly occupational, no additional cases among 97 household contacts of animal-exposed cases [R27].
- **No dispersion estimate was located**, and one cannot be estimated meaningfully from data in which nearly every case is zoonotic. Heterogeneity in H5N1 exposure (culling events, dairy milking) is animal-to-human exposure heterogeneity and should not be read as human superspreading.
- **Biological drivers:** no human infectiousness data (viral load to onward transmission) were found; the question is not testable without human-to-human transmission to observe.
- **Current situation (web sources, not independently re-fetched):** WHO Western Pacific weekly avian influenza updates through July 2026 state that no sustained human-to-human transmission has been identified with recent human H5 infections, and report a global cumulative total of 1,000 confirmed human H5N1 cases from 25 countries as of 7 July 2026.
- **Implication:** any statement about H5N1 superspreading is hypothetical. Models should treat it as a scenario, parameterised from other respiratory viruses, and flagged as such.

## 7. Graded conclusions

Grades apply to the claim in each cell, not to the pathogen overall. "Mixed" is the right default where evidence exists but cannot discriminate.

| Question | SARS-CoV-2 | Nipah | H5N1 (human-to-human) |
|---|---|---|---|
| Realised transmission is overdispersed (skewed offspring) | **Supported.** Many datasets, k below 1 in 93% of 28 studies [R7]; magnitude uncertain and setting dependent [R6] | **Mixed.** Consistent signal (7% transmit [R18]; 92% non-transmitters [R22]) but tiny counts, subcritical, outbreak-driven ascertainment | **Weak (not assessable).** Almost no human chains; no dispersion estimate located [R26, R28] |
| Behavioural: contact and mobility of infectious people | **Mixed.** Exposure duration and closeness matter [R5]; social settings exceed households [R4]; no persistent individual trait demonstrated; confounded with setting | **Mixed.** Caregiver exposure intensity (duration, body fluids, spouses) predicts infection [R21]; mobility of cases probably limited by severity (inference) | **Weak.** Only anecdotal caregiving chains [R24] |
| Biological: higher shedding, longer shedding, aerosol emission | **Mixed.** Large emission and shedding heterogeneity [R13, R15, R16]; modest realised-transmission gradient [R14]; short infectious window [R17]; no link from measured emission to offspring | **Weak.** Clinical proxies only (breathlessness, age) [R18, R21]; no human shedding data found | **Weak.** No human infectiousness data located |
| Setting and events: venue, crowding, ventilation, duration | **Supported.** Several designs converge [R4, R6, R10, R11, R30]; reporting bias and confounding with attendee biology remain | **Mixed.** Hospital and burial transmission documented [R19, R20]; case-based, frequency unknown | **Weak.** Occupational animal settings drive cases [R27]; no human-to-human setting data |
| Superspreaders as a discrete, persistent, biologically distinct class of people | **Weak.** No direct evidence of a distinct class; heterogeneity looks continuous [R2] | **Weak.** No evidence either way beyond the observation of few transmitters [R22] | **Weak (not assessable)** |

## 8. Where evidence is thin, contested or methodologically weak

1. **Attribution to individuals versus settings** is the central unresolved point for SARS-CoV-2; observational cluster data cannot separate them.
2. **Link from emission or viral load to realised offspring** has not been shown directly; the best linked data give a two-fold gradient [R14].
3. **k estimates** depend on method, period, data source and assumed distribution [R3, R7, R8]; early estimates [R3] used aggregated early data.
4. **Nipah:** all quantitative conclusions rest on a few dozen transmitting cases pooled from outbreak investigations; the one explicit superspreading model [R22] could only be read from its abstract here, and its k value was not verified. The systematic review itself states that estimates are scarce [R23].
5. **H5N1:** no human-to-human dispersion estimate exists in the sources screened. Estimates of R come from sparse household clusters [R26]. Anything stronger is speculation.
6. **Contested points:** whether the negative binomial is the right offspring distribution [R8]; whether very high k in households reflects biology or truncation [R6]; how much emission heterogeneity matters when viral infectiousness windows are short [R17].
7. **Reporting bias:** cluster literature over-represents notable events [R10].

## 9. Annotated reading list (30 items)

Every item below was found in PubMed or Crossref, its DOI resolves via doi.org, and its Crossref title and first authors were matched to the cited work; all abstracts were read. Full texts were not read for any item. Study design is stated for each.

**[R1] Woolhouse et al. (1997).** Heterogeneities in the transmission of infectious agents: Implications for the design of control programs. *Proceedings of the National Academy of Sciences*. https://doi.org/10.1073/pnas.94.1.338  
- Design: Cross-system empirical analysis of transmission-rate distributions. Pathogen: General.  
- Why it matters: Origin of the 20/80 pattern: across disease systems, about 20% of hosts contribute at least 80% of net transmission potential (R0). Framed around a stable 'core group' of hosts.  
- Limitations and cautions: Pre-dates the individual offspring (k) framework; the 20/80 here describes heterogeneity in transmission potential across host groups, not realised offspring counts.

**[R2] Lloyd-Smith et al. (2005).** Superspreading and the effect of individual variation on disease emergence. *Nature*. https://doi.org/10.1038/nature04153  
- Design: Theory plus statistical analysis of contact-tracing data from eight directly transmitted diseases. Pathogen: General.  
- Why it matters: Defines the individual-variation framework behind k. Individual infectiousness is highly skewed; models with this variation give more extinctions and rarer but more explosive outbreaks; individual-specific control beats population-wide control; proposes a formal definition of superspreading events.  
- Limitations and cautions: Estimates rest on contact-tracing data of mixed quality; the abstract verified here does not give the formal SSE threshold, which should be read in the paper.

**[R3] Endo et al. (2020).** Estimating the overdispersion in COVID-19 transmission using outbreak sizes outside China. *Wellcome Open Research*. https://doi.org/10.12688/wellcomeopenres.15842.3  
- Design: Modelling inference (negative-binomial branching process fitted to outbreak sizes outside China). Pathogen: SARS-CoV-2.  
- Why it matters: Early k estimate: about 0.1 (95% CrI 0.05-0.2 at R0 = 2.5), i.e. roughly 10% of cases causing 80% of transmission.  
- Limitations and cautions: Uses aggregated WHO situation-report counts of imported and local cases from early 2020; the upper bound of R0 was not informed by the data. Versioned DOI (v3); later versions exist.

**[R4] Adam et al. (2020).** Clustering and superspreading potential of SARS-CoV-2 infections in Hong Kong. *Nature Medicine*. https://doi.org/10.1038/s41591-020-1092-0  
- Design: Contact-tracing cohort and cluster analysis (1,038 cases, Hong Kong, Jan-Apr 2020). Pathogen: SARS-CoV-2.  
- Why it matters: 51 local clusters (309 cases) with 4-7 superspreading events; 19% (95% CI 15-24%) of cases seeded 80% of local transmission; social settings produced more secondary cases than households after age adjustment (P = 0.002).  
- Limitations and cautions: One city under intensive tracing and early NPIs; cluster detection depends on tracing intensity.

**[R5] Sun et al. (2021).** Transmission heterogeneities, kinetics, and controllability of SARS-CoV-2. *Science*. https://doi.org/10.1126/science.abe2424  
- Design: Contact-tracing cohort plus transmission modelling (Hunan, China). Pathogen: SARS-CoV-2.  
- Why it matters: 80% of secondary infections traced to 15% of primary infections; transmission risk rises with exposure duration and closeness of contact, and is modulated by demographic and clinical factors.  
- Limitations and cautions: Retrospective tracing data in a setting with strong interventions; transmission-risk covariates are partly confounded with setting.

**[R6] Chen et al. (2025).** Investigating setting-specific superspreading potential and generation intervals of COVID-19 in Hong Kong. *Nature Communications*. https://doi.org/10.1038/s41467-025-60591-x  
- Design: Cluster analysis, setting-stratified (8,647 cases, Hong Kong 2020-2021). Pathogen: SARS-CoV-2.  
- Why it matters: Dispersion varies by setting: k about 0.05 in retail (most heterogeneous) up to about 1.1 in households (least); mean expected new infections per cluster ranged 0.4-7.1 across eight settings.  
- Limitations and cautions: Single city; setting labels are coarse; household k is also capped by household size, so a near-1 value does not by itself show absence of individual variation.

**[R7] Wegehaupt et al. (2023).** Superspreading, overdispersion and their implications in the SARS-CoV-2 (COVID-19) pandemic: a systematic review and meta-analysis of the literature. *BMC Public Health*. https://doi.org/10.1186/s12889-023-15915-1  
- Design: Systematic review (28 peer-reviewed studies). Pathogen: SARS-CoV-2.  
- Why it matters: Mean k estimates ranged 0.04 to 2.97 across studies; 93% of studies reported mean k below 1. Control recommendations centred on preventing superspreading events.  
- Limitations and cautions: Wide k range reflects different data, periods, definitions and estimators; the review pools reported estimates rather than re-estimating on common data.

**[R8] Kremer et al. (2021).** Quantifying superspreading for COVID-19 using Poisson mixture distributions. *Scientific Reports*. https://doi.org/10.1038/s41598-021-93578-x  
- Design: Simulation study plus reanalysis of Hong Kong, India and Rwanda data. Pathogen: SARS-CoV-2.  
- Why it matters: Shows variance estimates can be biased under strong heterogeneity and that the offspring distribution chosen matters; a Poisson-lognormal fitted two of the three datasets better than the negative binomial.  
- Limitations and cautions: Only three datasets; shows sensitivity of k to distributional assumption rather than a new k.

**[R9] Endo et al. (2021).** Implication of backward contact tracing in the presence of overdispersed transmission in COVID-19 outbreaks. *Wellcome Open Research*. https://doi.org/10.12688/wellcomeopenres.16344.3  
- Design: Modelling study (branching process, contact tracing). Pathogen: SARS-CoV-2.  
- Why it matters: With overdispersion, backward tracing finds a primary case generating 3-10 times more infections than a random case. Illustrates a selection effect inherent in how clusters are found.  
- Limitations and cautions: Simple branching-process model; versioned DOI (v3).

**[R10] Leclerc et al. (2020).** What settings have been linked to SARS-CoV-2 transmission clusters?. *Wellcome Open Research*. https://doi.org/10.12688/wellcomeopenres.15889.2  
- Design: Systematic review of reported clusters, including media reports. Pathogen: SARS-CoV-2.  
- Why it matters: Clusters linked to a wide range of mostly indoor settings; few from schools, many from households, increasing numbers in hospitals and elderly-care settings.  
- Limitations and cautions: Authors note recall bias, biased media reporting and missing data; counts of clusters by setting are not rates. Versioned DOI (v2).

**[R11] Miller et al. (2020).** Transmission of SARS‐CoV‐2 by inhalation of respiratory aerosol in the Skagit Valley Chorale superspreading event. *Indoor Air*. https://doi.org/10.1111/ina.12751  
- Design: Outbreak investigation plus airborne-transmission modelling (Skagit Valley Chorale). Pathogen: SARS-CoV-2.  
- Why it matters: 53 of 61 attendees confirmed or strongly suspected cases after one rehearsal; inferred emission of about 970 +/- 390 infectious quanta per hour; risk halves with faster aerosol loss (ventilation) and a shorter event.  
- Limitations and cautions: Inference conditional on a single index case and aerosol dominance; cannot separate an unusually high-emitting individual from a favourable setting.

**[R12] Goyal et al. (2021).** Viral load and contact heterogeneity predict SARS-CoV-2 transmission and super-spreading events. *eLife*. https://doi.org/10.7554/eLife.63537  
- Design: Modelling inference linking viral-load kinetics to offspring distributions. Pathogen: SARS-CoV-2.  
- Why it matters: Superspreading events arise when an infector sheds at very high viral load and has many exposed contacts; the higher superspreading tendency relative to influenza is not explained by longer shedding. Directly relevant to separating the two explanations.  
- Limitations and cautions: Model-based, relies on published viral-load data; it posits an interaction rather than directly observing it.

**[R13] Ke et al. (2022).** Daily longitudinal sampling of SARS-CoV-2 infection reveals substantial heterogeneity in infectiousness. *Nature Microbiology*. https://doi.org/10.1038/s41564-022-01105-z  
- Design: Longitudinal viral-load cohort with mechanistic modelling (60 people, daily sampling up to 14 days). Pathogen: SARS-CoV-2.  
- Why it matters: Substantial person-to-person variation in infectious virus shedding, interpreted as contributing to superspreading; viral loads of Alpha and earlier lineages were mostly indistinguishable.  
- Limitations and cautions: Small cohort; infectiousness inferred from shedding, not from observed transmission.

**[R14] Marks et al. (2021).** Transmission of COVID-19 in 282 clusters in Catalonia, Spain: a cohort study. *The Lancet Infectious Diseases*. https://doi.org/10.1016/S1473-3099(20)30985-3  
- Design: Cohort of index cases and contacts (282 clusters, 753 contacts, Catalonia 2020). Pathogen: SARS-CoV-2.  
- Why it matters: Secondary attack rate 17%; 12% when index viral load was low versus 24% when high (adjusted OR 1.3 per log increase, 95% CI 1.1-1.5); household contact raised risk (OR 3.0). Viral load matters but moves risk by roughly two-fold, not orders of magnitude.  
- Limitations and cautions: Early-pandemic cohort drawn from a trial; limited to non-hospitalised adults; contact exposure not fully standardised.

**[R15] Chen et al. (2021).** Heterogeneity in transmissibility and shedding SARS-CoV-2 via droplets and aerosols. *eLife*. https://doi.org/10.7554/eLife.65774  
- Design: Systematic review and meta-analysis of respiratory viral loads with infectiousness modelling. Pathogen: SARS-CoV-2.  
- Why it matters: Intrinsic case variation in respiratory viral load facilitates greater overdispersion for SARS-CoV-2 than for 2009 H1N1; many cases pose minimal risk while a few shed tens to thousands of virions per minute by droplets and aerosols.  
- Limitations and cautions: Infectiousness is modelled from viral load, not observed in transmission; viral load is not equivalent to viable, transmitted virus.

**[R16] Edwards et al. (2021).** Exhaled aerosol increases with COVID-19 infection, age, and obesity. *Proceedings of the National Academy of Sciences*. https://doi.org/10.1073/pnas.2021830118  
- Design: Observational cohort (194 healthy humans) plus experimental infection (8 non-human primates). Pathogen: SARS-CoV-2.  
- Why it matters: Exhaled aerosol varied by about three orders of magnitude between people; 18% of subjects (35) produced 80% of exhaled bioaerosol, a 20/80-like pattern for emission itself.  
- Limitations and cautions: Mostly healthy volunteers, so it demonstrates emission heterogeneity, not that high emitters infect more people; link to infection came from animals.

**[R17] Cevik et al. (2021).** SARS-CoV-2, SARS-CoV, and MERS-CoV viral load dynamics, duration of viral shedding, and infectiousness: a systematic review and meta-analysis. *The Lancet Microbe*. https://doi.org/10.1016/S2666-5247(20)30172-5  
- Design: Systematic review and meta-analysis (79 SARS-CoV-2 studies). Pathogen: SARS-CoV-2 (with SARS-CoV, MERS-CoV).  
- Why it matters: Upper-respiratory viral load peaks in the first week; no study detected live virus beyond day 9 despite persistent RNA. Limits how long any individual can be 'highly infectious'.  
- Limitations and cautions: Pooled over heterogeneous assays and cohorts; RNA duration is a poor proxy for infectious duration.

**[R18] Luby et al. (2009).** Recurrent Zoonotic Transmission of Nipah Virus into Humans, Bangladesh, 2001-2007. *Emerging Infectious Diseases*. https://doi.org/10.3201/eid1508.081237  
- Design: Outbreak investigations, descriptive case series (Bangladesh 2001-2007). Pathogen: Nipah.  
- Why it matters: 23 introductions and 122 cases; 62 (51%) followed close contact with another case, but only 9 (7%) of cases transmitted; those with difficulty breathing transmitted more often (12% vs 0%, p = 0.03).  
- Limitations and cautions: Small counts; cases found through encephalitis investigations; clinical correlates are proxies for shedding.

**[R19] Gurley et al. (2007).** Person-to-Person Transmission of Nipah Virus in a Bangladeshi Community. *Emerging Infectious Diseases*. https://doi.org/10.3201/eid1307.061128  
- Design: Outbreak investigation with case-control study (Faridpur 2004). Pathogen: Nipah.  
- Why it matters: 36 cases, 33 with close contact with another case; contact with one specific patient carried highest risk (OR 6.7); Nipah RNA detected on hospital surfaces.  
- Limitations and cautions: Single outbreak; exposure assessed retrospectively.

**[R20] Sazzad et al. (2013).** Nipah Virus Infection Outbreak with Nosocomial and Corpse-to-Human Transmission, Bangladesh. *Emerging Infectious Diseases*. https://doi.org/10.3201/eid1902.120971  
- Design: Outbreak investigation with case-control comparison (Faridpur 2010). Pathogen: Nipah.  
- Why it matters: 16 cases (14 deaths); documented nosocomial transmission and corpse-to-human transmission linked to burial preparation, plus a physician infected after unprotected examinations.  
- Limitations and cautions: Anecdotal-level exposure detail in one outbreak; shows setting and ritual effects, not their frequency.

**[R21] Nikolay et al. (2019).** Transmission of Nipah Virus - 14 Years of Investigations in Bangladesh. *New England Journal of Medicine*. https://doi.org/10.1056/NEJMoa1805376  
- Design: Pooled outbreak-investigation data with regression (Bangladesh 2001-2014). Pathogen: Nipah.  
- Why it matters: 248 cases, 82 acquired person to person; R = 0.33 (95% CI 0.19-0.59); R higher with older case age and difficulty breathing (1.1, 95% CI 0.4-3.2 for age 45+ with breathlessness); 0 infections in 1,863 asymptomatic contacts; infection risk rose with exposure over 48 hours (adjusted OR 13) and body-fluid exposure (4.3).  
- Limitations and cautions: R is a mean for a subcritical process; wide interval for the highest-risk group; no viral-load data.

**[R22] Pan et al. (2026).** Superspreading potential of Nipah virus disease despite self-limited human-to-human transmission in community. *International Journal of Infectious Diseases*. https://doi.org/10.1016/j.ijid.2026.108683  
- Design: Modelling study of pooled historical outbreak reports (23 studies; 33 primary and 152 secondary cases). Pathogen: Nipah.  
- Why it matters: Estimated that 92.0% (95% CrI 87.0-95.4) of cases could not generate secondary cases; with 5 imported seeds, 1.1% (95% CrI 0.3-2.3) chance of an outbreak above 50 cases. The only quantitative study located that explicitly addresses Nipah superspreading potential.  
- Limitations and cautions: Based on abstract only (full text not accessible in this run); k value not verified here; pooled reports from Bangladesh and India, vulnerable to reporting bias towards larger clusters.

**[R23] Naidoo et al. (2026).** A systematic review of Nipah virus disease epidemiological parameters, outbreaks, and mathematical models. *The Lancet Infectious Diseases*. https://doi.org/10.1016/S1473-3099(26)00239-2  
- Design: Systematic review (119 papers, 243 parameters). Pathogen: Nipah.  
- Why it matters: Transmission parameter estimates are scarce; four of five central R estimates were below 1; only 8 of 39 mathematical models were fitted to data.  
- Limitations and cautions: States the evidence gap directly; based on abstract only.

**[R24] Ungchusak et al. (2005).** Probable Person-to-Person Transmission of Avian Influenza A (H5N1). *New England Journal of Medicine*. https://doi.org/10.1056/NEJMoa044021  
- Design: Cluster investigation (family cluster, Thailand 2004). Pathogen: H5N1.  
- Why it matters: Probable person-to-person transmission from a child to her mother after 16-18 hours of unprotected nursing, and to an aunt; no further chains identified.  
- Limitations and cautions: Three patients; single chain; non-sustained.

**[R25] Kandun et al. (2006).** Three Indonesian Clusters of H5N1 Virus Infection in 2005. *New England Journal of Medicine*. https://doi.org/10.1056/NEJMoa060930  
- Design: Cluster investigations (three family clusters, Indonesia 2005). Pathogen: H5N1.  
- Why it matters: Eight patients in three family clusters, most living in the same home.  
- Limitations and cautions: Tiny sample; cannot distinguish common animal exposure from human transmission.

**[R26] Aditama et al. (2012).** Avian Influenza H5N1 Transmission in Households, Indonesia. *PLoS ONE*. https://doi.org/10.1371/journal.pone.0029971  
- Design: Household final-size modelling of outbreak-investigation data (139 outbreaks, Indonesia 2005-2009). Pathogen: H5N1.  
- Why it matters: Household secondary attack rate 5.5%; human-to-human reproduction number 0.1-0.25 with upper bounds below 0.4; strong support for human transmission only when one large cluster was included.  
- Limitations and cautions: Retrospective, surveillance-detected outbreaks; exposure to poultry cannot be fully separated from human spread.

**[R27] Garg et al. (2025).** Highly Pathogenic Avian Influenza A(H5N1) Virus Infections in Humans. *New England Journal of Medicine*. https://doi.org/10.1056/NEJMoa2414610  
- Design: Case series with contact follow-up (46 US cases, Mar-Oct 2024). Pathogen: H5N1.  
- Why it matters: Cases were mild and mostly occupational (dairy or poultry); no additional cases among 97 household contacts of animal-exposed cases; no evidence of human-to-human transmission.  
- Limitations and cautions: Surveillance-detected cases, ascertainment biased to symptomatic or monitored workers; 2024 only.

**[R28] Ward et al. (2025).** Estimates of epidemiological parameters for H5N1 influenza in humans: a rapid review. *BMC Infectious Diseases*. https://doi.org/10.1186/s12879-025-11933-z  
- Design: Rapid review (46 articles) plus branching-process simulation. Pathogen: H5N1.  
- Why it matters: H5N1 transmissibility is low (R below 0.2) with longer incubation (about 4 days) and serial interval (about 6 days) than seasonal subtypes; key gaps remain on latent and infectious periods.  
- Limitations and cautions: Rapid rather than full systematic review; R mostly from sparse cluster data; no dispersion estimate.

**[R29] Sneppen et al. (2021).** Overdispersion in COVID-19 increases the effectiveness of limiting nonrepetitive contacts for transmission control. *Proceedings of the National Academy of Sciences*. https://doi.org/10.1073/pnas.2016623118  
- Design: Agent-based modelling study. Pathogen: SARS-CoV-2 (model).  
- Why it matters: Agent-based model with gamma-distributed individual infectivity (dispersion about 0.1) and close, regular and random contact sectors; when k is low, cutting random contacts outperforms cutting regular contacts.  
- Limitations and cautions: Stylised network; infectivity heterogeneity is assumed, so results illustrate implications not evidence of the cause.

**[R30] Chang et al. (2020).** Mobility network models of COVID-19 explain inequities and inform reopening. *Nature*. https://doi.org/10.1038/s41586-020-2923-3  
- Design: Modelling study with mobile-phone mobility networks (10 US metropolitan areas, 98 million people, points of interest). Pathogen: SARS-CoV-2 (model).  
- Why it matters: A small minority of 'superspreader' points of interest account for most infections in the model; occupancy caps work better than uniform mobility reduction. Closest published analogue to a POI-to-agent schedule design.  
- Limitations and cautions: Model predictions, not observed transmission; mobility data from a US phone panel.


## 10. Method, search log and limits

- **Sources searched:** PubMed (via the PubMed connector) for SARS-CoV-2, Nipah and H5N1 transmission, overdispersion and superspreading (about 25 queries; Nipah queries returned very few relevant hits), plus Crossref for DOI verification and metadata. Two web searches were run only to check the current situation for H5N1 and Nipah (West Bengal, January 2026).
- **Not a registered systematic review.** The search was bounded to a single run; it was not exhaustive and did not cover Embase, Web of Science or preprint servers. Items in the list were chosen for design diversity and verifiability, not by a quality-scoring procedure. "Not found" statements mean not found in the sources screened.
- **Citation discipline:** DOIs were taken only after Crossref or PubMed returned a matching record. Several DOIs recalled from memory at the start failed verification (wrong work or no record) and were corrected or dropped; none of those was used.
- **Papers noted but not included in the 30 (DOIs verified in Crossref during the run, abstracts partly read, not annotated):**
  - Lau et al. 2020, PNAS (individual-level spatiotemporal model, Georgia): https://doi.org/10.1073/pnas.2011802117
  - Yang et al. 2021, PNAS (viral-load 'supercarriers', Boulder): https://doi.org/10.1073/pnas.2104547118
  - Lemieux et al. 2021, Science (Boston phylogenetics and superspreading events): https://doi.org/10.1126/science.abe3261
  - Coleman et al. 2021, Clin Infect Dis (viral load in respiratory aerosols): https://doi.org/10.1093/cid/ciab691
  - Adenaiye et al. 2022, Clin Infect Dis (infectious virus in exhaled aerosol): https://doi.org/10.1093/cid/ciab797
  - Clayton et al. 2012, Emerg Infect Dis (Nipah strain shedding in ferrets): https://doi.org/10.3201/eid1812.120875
- **No claim here comes from unpublished partner results or background.**
