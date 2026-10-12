# ACT-049: further experiments worth running, and what is already done

12 October 2026. Written after the D1 and D2 results. Each item says what claim it defends, what would change if it came out the other way, and roughly what it costs. Items are ordered by how much they change the paper, not by how easy they are. No PanSim runs were made for this note; the power and resolution figures below were computed here.

## Summary

Three experiments are worth running (X1 to X3). Two more are worth running only under a condition that is named (X4, X5). Three things that look like experiments are better handled as analysis or as reporting (A1 to A3). One question cannot be answered without a change to what the simulator writes out (X6).

If only one thing is run, run X1: it is the only item that could overturn a claim already in the draft.

## X1. Does the tail result survive venue subdivision? (highest value)

**Claim defended.** Section 5.7 of the draft, one of the four results in the paper's spine: matching dispersion does not match tail risk, because offspring are correlated with the setting the parent was infected in.

**The problem.** The effect arises in a model where 17 hospital locations serve the whole city, each with about 430 occupants at peak. The draft already says this, but a reviewer will ask whether the result is a property of transmission structure or of venue aggregation. At present we cannot answer.

**Design.** Split the hospital, nursing-home and health-centre locations into virtual sub-units (wards) of a plausible size, say 20 to 40 occupants, preserving total capacity, total visits and each agent's assignment, so that only co-presence changes. Re-run the two subcritical chain arms from phase 3 at matched all-node mean offspring, about 38,600 chains per arm as before. Compare: chains above 10 and above 50, maximum chain size, the departure from a negative binomial with the arm's own mean and dispersion, and the hospital-to-hospital offspring correlation that drives the effect.

**What each outcome means.** If the heavy extreme and the thin middle survive subdivision, setting-correlated offspring is a transmission-structure result and the claim stands as written. If it vanishes, the claim must be restated as an artefact of aggregated venues, and Section 5.7 becomes a caution about model construction rather than a finding. Either way the paper improves.

**Cost.** One input-file transformation plus two chain arms. No new mechanism. Cheap relative to what it protects.

## X2. The lever reversal at matched epidemic size, and the household-share gradient

**Claim defended.** Section 5.8, a second spine result: intervention ranking tracks where transmission happens, not dispersion.

**Two weaknesses.** First, the two arms differ 2.6-fold in uncontrolled final size (54,784 against 20,928), so only within-arm orderings are safe and the reversal could in principle be a saturation effect. Second, the household-share predictor that explains the reversal rests on five points drawn from three different experiments, with one rank inversion, and is currently labelled exploratory.

**Design.** Calibrate the person-level and setting-level arms to the same early reproduction number **and** the same uncontrolled final size, by adjusting the coefficient after the mechanism is set, and repeat the four levers. Separately, run three arms chosen to land at household shares near 0.25, 0.37 and 0.45, measuring the household share of transmission in the same runs rather than importing it. Pre-register the prediction already on record: the difference between venue closure and household isolation changes sign near a household share of 0.37.

**What it buys.** If the reversal holds at matched final size, the saturation objection is closed. If the three new arms fall on the predicted gradient, the predictor moves from a supplementary figure to a result, and the paper can state a decision rule: estimate the share of transmission in closable settings, because dispersion will not tell you which lever wins.

**Cost.** Two calibrated arms times five levers, plus three calibration runs and their levers. The largest item on this list.

## X3. Chain-size calibration for the subcritical scenarios

**Claim defended.** The recommendation in the discussion that for subcritical pathogens one should calibrate to the chain-size distribution rather than to a dispersion parameter. At present the paper recommends this without demonstrating it.

**Design.** Take the Nipah-like target from the literature: reproduction number 0.33 with about 92% of cases infecting nobody, which corresponds to a dispersion near 0.036. Calibrate one arm to the dispersion value and a second to the zero-offspring share and the chain-size distribution, and compare both against the published outbreak-size pattern, including the roughly 1% chance of an outbreak above 50 cases from five seeds. About 38,000 chains per arm as in phase 3.

**What it buys.** A worked demonstration of the paper's own recommendation, and a direct test of whether this model can reach the Nipah signature at all. Phase 3 suggested it cannot without an imposed multiplier; if that holds under a proper chain-size calibration it is a clean statement of a model limitation rather than a claim about Nipah.

**Caveat to carry.** The reproduction number and the zero share come from different studies of different outbreak sets, so the joint target is an approximation and should be presented as one.

## X4. Sharper infectiousness profile (conditional)

**Run only if** the paper claims the lever ranking is general rather than specific to this natural history. The current draft explicitly does not, so this is optional.

The PS2 null is well-founded but narrow: the half-maximum window spans days 1 to 19 and holds 95% of the infectiousness area, so "peak mode" was nearly the constant multiplier, and only 23% of infectiousness falls before the median diagnosis. The one arm with a genuinely narrow window showed the signature the experiment was looking for, with the offspring-to-generation-interval correlation rising from about 0.10 to 0.38. A single arm with a compressed profile would show whether the lever ranking is stable when timing actually bites.

## X5. Lognormal arm recalibration (small, conditional)

Section 5.4 states that the dispersion floor under coupling is specific to the gamma family, citing a lognormal arm at 0.464 against the gamma 0.954. That arm's twelve-replicate early reproduction number is 1.54 against its calibrated 1.45, so the comparison is not at matched growth. One recalibrated arm would make the family claim clean. Worth doing because the floor is an argument the paper makes twice.

## A1. Equivalence power: report it, do not re-run

The observational-equivalence result (Section 5.3) is the paper's cleanest claim and it is a null, so it needs a stated resolution. From the delivered intervals, the difference in early dispersion between the per-episode and persistent arms is +0.016 with a 95% interval of -0.022 to +0.054. **The arms are therefore equivalent to within about plus or minus 13% of the dispersion value, not exactly equivalent.** Tightening that to plus or minus 5% would need about 76 replicates, which is not worth it. The right action is to state the bound in the paper rather than run more. The same should be done for final size, which needs the per-replicate spread that was not reported.

## A2. The tail claim rests on shape, not on counts

For the chain result, 17 chains above 50 in one arm against 19 in the other is not a difference (Fisher exact p = 0.87), and detecting a doubling of that probability would need about 52,000 chains per arm. The claim should rest, as the draft now does, on the departure from the negative binomial: 236 chains above 10 observed against about 426 expected, which is overwhelming. No further runs are needed, but the draft and any talk slide should avoid quoting "17 against 19" as evidence of a difference between arms.

## A3. What a field study could ever resolve

This is an analysis, not a run, and it belongs in the discussion. Simulating a contact-tracing study at a true dispersion of 0.3 gives a 5th-to-95th-percentile range of 0.20 to 0.46 from 100 index cases, 0.22 to 0.41 from 200, and 0.24 to 0.36 from 500. **A typical tracing study cannot separate a dispersion of 0.2 from one of 0.5.** This bounds the whole attribution question from the data side, independently of any model, and it explains why the published range of 0.1 to 0.6 is as wide as it is. It is a strong addition to the discussion and costs nothing.

## X6. Contact availability: blocked by what the trace stores

The remaining open question from D1 is whether contact availability limits the largest multipliers, which would explain the tail bend. It cannot be answered from existing output, because the transmission trace stores only infected agents, so uninfected contacts are invisible and the distribution of distinct contacts per infectious episode is unmeasurable. Answering it needs the simulator to emit, per infectious episode, the number of distinct people co-present. That is a change to the output, not an experiment. Worth requesting only if the paper wants to explain the bend rather than report it.

## What is already settled and should not be re-run

- Whether occupancy can generate overdispersion in this kernel. Settled negatively by algebra and by the density-exponent runs.
- Whether a persistent trait differs from a per-episode multiplier. Settled, within the bound in A1.
- Whether the contact structure absorbs the model's own heterogeneity. Tested and rejected by D1; the composition rule is retired and should not be re-fitted.
- Whether detection inflates the estimate. Settled in direction across every sampling scheme tested.
- Venue crowding, capacity and explicit events. Parked by decision until the national and regional rebuild lands, and the individual-offspring correction in D2 lowered the urgency, because the model-literature gap on setting dispersion is narrower than phases 1 to 3 suggested.

## Suggested order

1. X1, because it can overturn a published claim.
2. A1, A2 and A3, which are writing and analysis and can happen in parallel.
3. X5, one small arm.
4. X2, the largest item, worth it if the lever result is to carry weight.
5. X3, if the subcritical recommendation is to be demonstrated rather than asserted.
6. X4 and X6 only under the conditions named.
