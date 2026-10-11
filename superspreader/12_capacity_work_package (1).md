# ACT-049 follow-on work package: location capacity audit and recalculation

Hand-off document. Written 10 October 2026 in response to the note that `locations0.json` has a capacity field, that for residences it follows the number of assigned people, and that for other types the values may not make sense.

Scope: fix capacity and area first, then run the volume-model experiments from 11_volume_model_proposal.md. Sections 1 to 5 are the capacity work. Section 6 is the experiment sequence that follows it.

Everything below is a specification to execute, not work already done. No PanSim runs were made for this document. The per-type numbers in section 4 are my estimates, not sourced values, and section 4.4 says what to replace them with.

## 1. Why capacity has to be fixed before the experiments, not alongside them

**Capacity is currently inert, which makes it safe to change.** Phase 1 established that the kernel has no hard occupancy limit, and the hazard divides by head count present, not by capacity. So capacity today affects nothing in the transmission path. It can be rewritten without invalidating any existing fit, including the ACT-047 calibration. That window closes the moment the volume model lands, because capacity then enters the denominator of every infection.

**Capacity error is not neutral noise, it manufactures dispersion.** If effective volume carries a multiplicative error, that error is one fixed draw per location and applies to every visit an agent makes to it. Because agents revisit the same few places, home, workplace or school, the error is correlated across an infector's visits and does not average out. That makes it mathematically the same object as genuine venue heterogeneity: a per-location multiplier on the per-visit hazard.

I tested this in a toy model: each infector splits its infectious period over 5 distinct locations with Dirichlet time shares, real per-visit variation is present (lognormal, sigma 0.8), R is 1.4, and the capacity error is one lognormal draw per location.

| capacity error, 90% spread | resulting k | relative to no error |
|---|---:|---:|
| none | 2.00 | 1.00 |
| 3x | 1.77 | 0.89 |
| 10x | 1.07 | 0.54 |
| 27x | 0.61 | 0.31 |
| 139x | 0.19 | 0.10 |

A 27-fold spread in capacity error cuts k to a third of its error-free value. The baseline k of 2.0 here is a property of this toy structure, not a PanSim figure, so read the ratios rather than the levels.

**The correlation is what does the damage, and it depends on how concentrated an agent's locations are.** Redrawing the same error independently at every visit instead of once per location leaves k at 1.82 rather than 0.61, because independent draws average away. Varying the number of distinct locations per infector at a 27-fold spread:

| distinct locations per infector | k without error | k with error | ratio |
|---|---:|---:|---:|
| 2 | 1.16 | 0.32 | 0.28 |
| 5 | 1.97 | 0.59 | 0.30 |
| 10 | 2.87 | 1.08 | 0.38 |
| 30 | 4.05 | 2.16 | 0.53 |

Even at 30 distinct locations, where averaging is strongest, capacity error still halves k. PanSim agents concentrate most of their time in a home plus one work or school location, which is the left end of this table and the worst case. So if capacities are wrong by an order of magnitude in either direction, the volume-model experiment will report a dispersion reduction and we will not know whether it came from ventilation physics or from bad capacity data. This is the argument for doing sections 2 to 5 first and as a separate deliverable.

**The schema already anticipated this.** The original PanSim location input specification (Drive: "Locations" and "Locations input data", 2020) defines per location: location ID, x, y, **area in square meters**, type, average visitor number per hour, **maximal capacity**, and initial operation state. The area column carries the comment "agent density potentially affects infection probability". So area and capacity were designed as separate fields with exactly this use in mind. The task is to populate them consistently, not to invent a new field.

## 2. Phase A: audit what is there now

Produce one table, one row per location, with: ID, type, current capacity, current area if populated, number of agents assigned, and occupancy statistics from an existing infection trace. Nothing here needs a GPU or a new simulation; phase 2 already added `occupancy` and `infectiousPresent` trace columns, and the schedule assignment gives the rest.

Diagnostics to compute per type:

1. **Coverage.** How many locations have a non-null, positive capacity. Same for area. A type where area is unpopulated has to be handled by rule, not by data.
2. **Capacity distribution.** Count, min, median, 90th percentile, max, and the ratio of max to median. A type whose capacity is constant across all its locations is a placeholder, and should be flagged as such.
3. **Capacity against the type definition.** The type names carry intended ranges: type 5 is 20 to 100 people, type 6 is 100 to 1000, type 9 is 50 to 500. Report the fraction of locations of each type whose capacity falls inside its stated range. This is the single most informative check and it needs no external data.
4. **Capacity against assignment.** Ratio of assigned agents to capacity per location. For residences this is 1 by construction, which confirms the reported behaviour. For a workplace it should be near 1 if capacity means staff size. For a shop or restaurant it should be well above 1, because many more people visit than are assigned.
5. **Capacity against realised peak occupancy.** From the trace, per location, take the maximum concurrent head count over a representative uncontrolled week, and the 95th percentile of the daily maxima. Report the ratio of peak occupancy to capacity. **Any location where this exceeds 1 would produce a fill fraction above 1 under the volume model, which is the failure mode to find now rather than after calibration.** Report the fraction of locations per type above 1, and the worst offenders.
6. **Implied area per person.** Where area is populated, area divided by capacity. Compare against section 4. A classroom implying 40 square metres per pupil, or a dwelling implying 2, localises the error.
7. **Aggregate sanity.** Total capacity per type against the Szeged population of about 178,750 agents. Total residence capacity should be close to the population. Total classroom capacity should be close to the school-age population. Total hospital capacity should be a few per thousand, not a few per hundred.

Deliverable for phase A: a CSV of per-location diagnostics, a per-type summary table, and a short list naming which types are usable as they stand, which need rescaling, and which need rebuilding.

## 3. Phase B: decide what capacity means, before changing any number

This is a definition choice and it must be made explicitly, because the audit above will give different answers under each reading. Three candidates:

- **B1, design capacity.** The number of people the venue is built to hold, as a fire-code or seating figure. Matches the type descriptions and the "maximal capacity" column name. Independent of the simulated schedule, so it does not move when schedules change.
- **B2, typical concurrent occupancy.** The number usually present. Matches the "average visitor number in each hour" column, which is a separate field in the schema.
- **B3, assigned-member count.** What residences currently use.

**Recommendation: B1 for the capacity field, with B2 kept separately if it is wanted for validation.** Three reasons. The field is named maximal capacity. Fill fraction is only interpretable against a design maximum, since under B2 the mean fill fraction is 1 by construction and the crowding signal disappears. And B3 makes capacity a function of the schedule generator, so every ACT-004 change would silently alter transmission once the volume model is in.

Consequence to accept: under B1, residences change meaning. A dwelling assigned 4 people gets a design capacity reflecting the dwelling size, not the household size, so fill fraction stops being 1. Since household transmission is about 46% of infections in the flat baseline, this is the single most load-bearing change in the package. Section 6 keeps residences on the old behaviour in the first experiment so the effect is isolated.

## 4. Phase C: recalculation rules

### 4.1 The identity to maintain

Populate area and capacity so that they stay consistent with each other and with the ventilation table:

    capacity_loc = area_loc / m2_per_person[type]
    volume_loc   = area_loc * height[type]
    effective_removal_loc = volume_loc * (ACH[type] + L)      with L about 1 per hour

Populate whichever of area or capacity is better evidenced for a given type and derive the other. Never set both independently, or the volume model and any occupancy cap will disagree.

### 4.2 Per-type parameters

These decompose the table already delivered in 11_volume_model_proposal.md into the area and height that the schema wants. The round trip is consistent to 0.2 cubic metres per person. **All values are my estimates from general building knowledge. None was looked up.**

| ID | Type | m2 per person | Height (m) | m3 per person | ACH low / mid / high |
|---|---|---:|---:|---:|---|
| 2 | Residence | 15.4 | 2.6 | 40 | 0.3 / 0.5 / 1.0 |
| 3 | School stay site (non-classroom) | 5.0 | 3.0 | 15 | 1.0 / 2.0 / 4.0 |
| 33 | Classroom | 2.0 | 3.0 | 6 | 0.7 / 1.5 / 4.0 |
| 4 | Standard workplace | 10.0 | 3.0 | 30 | 1.0 / 2.0 / 4.0 |
| 13 | Non-standard-schedule workplace | 10.0 | 4.0 | 40 | 1.0 / 2.0 / 4.0 |
| 44 | University | 2.9 | 3.5 | 10 | 1.0 / 3.0 / 6.0 |
| 5 | Restaurant / pub (20-100) | 1.9 | 3.2 | 6 | 1.0 / 3.0 / 6.0 |
| 6 | Cinema / theatre / concert (100-1000) | 1.2 | 5.0 | 6 | 3.0 / 6.0 / 10.0 |
| 7 | Short-visit shop / front office | 7.1 | 3.5 | 25 | 2.0 / 4.0 / 8.0 |
| 8 | Longer-visit shop / gym / playground | 8.6 | 3.5 | 30 | 1.0 / 3.0 / 8.0 |
| 9 | Church (50-500) | 1.9 | 8.0 | 15 | 0.3 / 1.0 / 2.0 |
| 10 | Outdoor recreation | n/a | n/a | 500 | 10.0 / 20.0 / 50.0 |
| 11 | Closed facility (prison) | 4.0 | 3.0 | 12 | 0.3 / 0.7 / 1.5 |
| 12 | Hospital | 8.3 | 3.0 | 25 | 2.0 / 4.0 / 6.0 |
| 14 | Health centre | 4.0 | 3.0 | 12 | 1.0 / 3.0 / 6.0 |
| 22 | Nursing home | 7.1 | 2.8 | 20 | 0.3 / 1.0 / 2.0 |

Type 1, public space, is omitted: it carries infectiousness 0 in `locations0.json`, so neither capacity nor volume changes anything for it. Type 10, outdoor, should be given an effective volume large enough that virus cannot accumulate, rather than a literal area.

### 4.3 Per-type recalculation recipe

Work in this order, because later types can be anchored to earlier ones.

1. **Residences (2).** Set area from the dwelling-size distribution for Hungary conditioned on household size, then capacity = area / 15.4. Keep the assigned-member count in a separate field. If no dwelling data is obtained, the defensible fallback is capacity = assigned members + 1, which at least varies with dwelling size and keeps fill fraction below 1.
2. **Classrooms (33) and school sites (3).** Classroom capacity is a class-size figure, not an area figure: set it from Hungarian class-size statistics, then derive area as capacity x 2.0. School stay sites are the shared non-classroom spaces, so anchor their capacity to the sum of the classroom capacities in that school.
3. **Workplaces (4, 13) and university (44).** Anchor capacity to the number of agents assigned to work there, which the synthetic population already fixes, then derive area. Capacity should exceed assignment modestly, suggest 1.1 to 1.3 times, to allow for visitors and non-simultaneous shifts. Type 13 should differ from type 4 in shift pattern, not in area per person.
4. **Venues with a stated range in the type name (5, 6, 9).** Sample capacity from a right-skewed distribution over the stated range rather than assigning the midpoint, because a lognormal over 20 to 100 and 100 to 1000 is itself a source of the venue heterogeneity the study is trying to represent. Suggest a lognormal fitted so the 5th and 95th percentiles sit at the stated bounds. Record the chosen distribution: it is a modelling assumption, and section 6 varies it.
5. **Shops and services (7, 8, 14).** Capacity from area if OpenStreetMap or a retail register gives floor areas for Szeged, otherwise from the per-person figures above with a plausible area distribution. These types have high throughput and low simultaneous occupancy, so the audit's peak-occupancy check matters most here.
6. **Hospital (12), nursing home (22), prison (11).** Capacity is a bed count plus staff, which is a published figure for Hungarian institutions, not something to infer from area. Derive area from capacity. These three types also most clearly violate the well-mixed assumption at the whole-building scale; see section 5.
7. **Outdoor (10).** No capacity needed in a transmission sense. Set effective volume to a large constant.

### 4.4 Replace the guesses with sources

Before any result from this work is reported outside the project, the section 4.2 values should come from sources rather than from my estimates. The ones normally used for this purpose are the European indoor-environment standard EN 16798-1 and the ASHRAE ventilation standard 62.1 for area and air-change figures by building category; KSH (Hungarian Central Statistical Office) for dwelling sizes, class sizes, hospital beds and nursing-home places; and OpenStreetMap for Szeged building footprints where a per-location area is wanted. **I have not retrieved or checked any of these in this session, and I cannot confirm what any of them contains.** Treat the list as a starting point for whoever executes this, not as a citation.

## 5. Phase D: validation and acceptance

The recalculated file is acceptable when all of these hold, and the check should be written as a script that runs on the file so it can be rerun after any regeneration:

1. Every location has positive capacity and positive area, and capacity equals area divided by the type figure to within rounding.
2. For every type with a stated range in its name, at least 90% of locations fall inside that range.
3. Realised peak concurrent occupancy from an uncontrolled reference run does not exceed capacity for more than a small, stated fraction of locations, suggest 1%. Fill fractions above 1 must be either impossible or explicitly clamped, and the clamping rule stated.
4. Total capacity per type is within a stated tolerance of the external aggregate: residences against population, classrooms against school-age population, hospital beds against the published Szeged figure.
5. The capacity distribution within a type has a documented shape, with the max-to-median ratio reported. A type that comes out constant has not been fixed, only relabelled.
6. A diff report against the old file: how many locations changed, by how much, and the resulting change in total capacity per type.

**Regression check.** Re-run one existing calibrated configuration with the new capacities and the volume model off. Because capacity is inert in the current kernel, the output must be bit-identical or within replicate noise. If it is not, something else reads capacity and that has to be found before proceeding.

## 6. Phase E: the experiments, in order

Only after phase D passes.

1. **E0, regression.** As above. Confirms capacity is inert pre-volume-model.
2. **E1, volume model on non-residential types only.** Add `--volumeModel off|nonresidential|all`, default off, table loaded from JSON so values can change without recompiling. Residences keep the head-count denominator. Recalibrate the infection coefficient to early R 1.39 using the phase 2 protocol, then report early k with bootstrap interval, v10, household share of infections, and the fill-fraction distribution at the moment of infection. Compare against the phase 2 baseline column, where early k is 5.25.
3. **E2, residences included.** Same, with `--volumeModel all`. The quantity to watch is the household share of infections, about 46% in the flat baseline. If it moves a lot, the dwelling-area data in section 4.3 step 1 is load-bearing and needs sourcing before anything else is concluded.
4. **E3, ventilation sensitivity.** Low, mid and high ACH columns from the table, plus a run with fill fraction pinned at 1, which separates the ventilation contrast from the fill-fraction variation. This also bounds how much of any k change is attributable to parameter guesswork.
5. **E4, capacity-distribution sensitivity.** Re-run E1 with the type 5, 6 and 9 capacity distributions set to their midpoints instead of the lognormal. If k differs materially, the venue-size distribution is doing the work rather than the physics, which is worth knowing and is cheap to test.
6. **E5, explicit events at fixed volume.** The only run that can show whether published-range k is reachable without the per-agent multiplier. Target the toy figure from 11: a small fraction of visits at a large hazard multiple, implemented as a full, long, poorly ventilated gathering rather than as an abstract multiplier.
7. **E6, wave-2 hold-out.** Refit to the wave-1 hospital area and compare wave 2 against the baseline's 222, as in phase 2 section 7. Only a configuration that survives this is a production candidate. The current survivor is the shape-0.3 multiplier at early k 0.32.
8. **E7, intervention levers.** Repeat the Set I and Set II comparison with a volume-model Set II, now including two levers that are near no-ops today: an occupancy cap, which binds because fill fraction enters, and a ventilation improvement applied per type. This is the experiment that would show whether the intervention ranking from phase 2 survives a physically grounded setting mechanism.

**Expected outcome, stated in advance so it can be wrong.** Based on the toy calculation in 11, I expect E1 to E4 to move early k from about 5 into the 2 to 4 range, not to 0.1, and I expect E5 to be the only route below 1 without the multiplier. If E1 alone produces k near 0.5, the first thing to suspect is residual capacity error of the kind quantified in section 1, not a physics success. Section 1 shows an error of that size is enough to produce exactly this result on its own.

## 7. Open questions for whoever executes this

1. Does anything other than the transmission kernel read capacity today, for example closure rules or the schedule generator? E0 answers this empirically, but a code check is faster.
2. Is area populated in `locations0.json` at all, or only present in the 2020 schema?
3. Are hospital, nursing home and prison single well-mixed locations, or are they subdivided into wards and rooms? At the whole-building scale the well-mixed assumption is worst exactly where the caregiving profile matters most, and the phase 2 long-chain finding came from those types.
4. Is there an existing Szeged building or POI dataset in the ACT-004 pipeline that already carries footprints, which would make section 4.3 steps 5 and 6 mostly a join rather than an estimation?
