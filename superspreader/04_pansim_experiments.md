# ACT-049 Proposed PanSim experiments

Four experiments, each with hypothesis, setup, outputs and dependencies. These are proposals only: no simulations or PanSim runs were made in this work packet. Citations [R#] refer to 01_literature_review.md. Hypotheses are stated so that they can fail.

## Experiment 1. Variance partition ablation

- **Question:** How much of model-generated overdispersion comes from agent multipliers, schedule heterogeneity and venue amplification, and how sensitive is it to the split?
- **Hypothesis:** Schedule heterogeneity plus venue amplification alone can produce k below 1 if venue occupancy is heavy-tailed, whereas a multiplier alone matches a target k but cannot reproduce setting concentration (a minority of venues responsible for most infections [R30]).
- **Setup:** Full factorial over (a) multiplier on or off with shape k_b in {0.1, 0.3, 1, infinity (off)}, (b) schedule heterogeneity on (ACT-004 schedules) or off (agents assigned to venues at random with matched marginal occupancy), (c) venue amplification on or off. Early epidemic, no interventions, SARS-CoV-2-like natural history, fixed seed count, many replicates per cell.
- **Outputs and metrics:** Realised offspring distribution and negative-binomial k estimate with interval; share of infections from top 10% and 20% of infectors and of venues; extinction probability; variance of early growth rate; fraction of transmission in households vs community venues.
- **Dependencies:** ACT-004 schedules for factor (b). Targets for comparison: [R3, R4, R6, R7].
- **Decision value:** Quantifies the sensitivity of the model's output to the identifiability gap described in the options document.

## Experiment 2. Same k, different mechanisms: intervention response

- **Question:** If two parameterisations are calibrated to the same marginal k but with different sources of heterogeneity, do they predict different intervention effectiveness?
- **Hypothesis:** A biology-dominated parameterisation favours individual-targeted measures (isolation, testing) and responds weakly to venue occupancy caps, whereas a setting-dominated parameterisation shows the opposite [R29, R30, R2].
- **Setup:** Pick two parameter sets matched to the same k (and R) over early generations: set I with a heavy multiplier and light schedule heterogeneity, set II with a light multiplier and heavy venue amplification. Apply (i) venue occupancy caps, (ii) random contact reduction, (iii) isolation of symptomatic cases with and without backward tracing [R9], at matched cost or coverage.
- **Outputs and metrics:** Cumulative infections and peak prevalence versus intervention; effective R over time; cases averted per unit of intervention; divergence between set I and set II.
- **Dependencies:** ACT-004 schedules; ACT-047 calibration for intervention implementation and baseline R in the Hungarian setting. Interventions must be implemented identically across parameter sets.
- **Decision value:** If predictions diverge materially, the attribution question matters for policy, and the project should prioritise data that separate the components. If not, the simpler model suffices.

## Experiment 3. Ascertainment-bias audit of k

- **Question:** How far do detection and tracing distort k as an analyst would estimate it from detected clusters, compared with the true simulated k?
- **Hypothesis:** Estimated k from detected clusters differs from true k in a direction and size that depend on the detection process (symptom-based, cluster-triggered, backward tracing), and ignoring this inflates the apparent superspreading of the detected tail [R9, R10, R8].
- **Setup:** Run baseline simulations with known transmission trees. Apply several observation models: symptom-driven testing with delay, cluster-triggered testing, forward-only and forward-plus-backward tracing. Estimate k from observed clusters using a negative-binomial model and a lognormal alternative [R8].
- **Outputs and metrics:** Bias and coverage of k and of the 80% share under each observation model; sensitivity to choice of offspring distribution.
- **Dependencies:** none beyond the base model; useful to run before experiment 1 so that comparisons to published k [R3, R4, R6, R7] are made on a like-for-like basis.
- **Decision value:** Sets how far published k can be trusted as a calibration target.

## Experiment 4. Subcritical stuttering chains in health-care and household settings (Nipah-like and H5N1-like scenarios)

- **Question:** For R below 1, how does heterogeneity in caregiving and health-care exposure intensity change the distribution of chain sizes and the chance of a large cluster?
- **Hypothesis:** At the same mean R, adding setting-based heterogeneity (long caregiver exposure, nosocomial settings, burial gatherings) raises the probability of rare large clusters and the share of cases with zero offspring compared with a homogeneous model [R21, R22, R20].
- **Setup:** Seed 1 to 5 index cases; set R about 0.3 for the Nipah-like scenario [R21] and R at or below about 0.25 for H5N1-like [R26, R28]; vary exposure intensity by setting, with and without a health-care setting layer; repeat with and without a modest multiplier. Compare chain-size distributions with qualitatively known patterns (many size-one chains, occasional clusters) rather than fitting.
- **Outputs and metrics:** Chain-size distribution; probability of a chain above 10 and above 50 cases; proportion of cases with zero offspring; effect of isolating index cases early.
- **Dependencies:** ACT-047 (health-care worker setting and interventions) and ACT-004 for household and hospital-like venues. No Hungarian outbreak data are implied; these are stylised scenarios.
- **Caveats:** The reference patterns come from Bangladesh and India with few cases [R18, R21, R22]; scenarios should be labelled as illustrative and not used for forecasting. For H5N1 the whole scenario is hypothetical because human-to-human k has not been estimated.
