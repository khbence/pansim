# ACT-049 proposal: approximate density with per-type effective volumes

Response to the suggestion of putting educated-guess volumes on location types. All numbers in the table are my estimates from general building knowledge, not sourced values: none was looked up in this session. Candidate sources to replace them with are listed in section 5. No PanSim runs were made; the dispersion figures in section 3 come from a toy calculation, not from PanSim.

## 1. What to implement

Replace the head-count denominator with a fixed room-size denominator. In the present kernel, density = sum(infectiousness present) / (people present). In a well-mixed room the concentration an occupant breathes depends on how much virus-laden air is removed per hour, not on how many people share it, so the proposal is:

    density = sum(infectiousness present) / ( Ncap_loc * q_type )
    q_type  = m3_per_person_type * (ACH_type + L)          [m3/h of effective removal per capacity-person]

with L about 1 per hour as a combined deposition and viral-inactivation loss rate (an assumption). Ncap_loc is the location's capacity, a per-location attribute, and q_type comes from the table below. The effective room size is then Ncap_loc * q_type, a fixed quantity per location, and the number of people present only enters through the sum in the numerator. Offspring per infector now scale with fill fraction and with time present, as they do in the quanta models used for the Skagit chorale [R11].

## 2. Two traps, one of which makes a per-type volume alone useless

**Trap 1: a volume per TYPE is degenerate with the existing venue profile.** If every location of a type gets the same volume, the denominator is a constant per type and is exactly absorbed by the existing per-type infectiousness multiplier (`--venueProfile`). Nothing new is expressed. The benefit comes only if volume is tied to a per-location quantity. The type descriptions suggest the generator already has one: restaurants 20 to 100 people, cinemas 100 to 1000, churches 50 to 500. I do not know whether `locations0.json` stores a capacity or area per location, which I asked earlier and which decides how cheap this is. If it does not, the number of agents assigned to the location is the fallback, with the caveat from phase 2 that assigned size is not presence.

**Trap 2: scaling volume with capacity cancels the crowding effect again, almost.** If V = v_per_person * Ncap, then offspring per infector per visit is proportional to N_present / V = fill fraction / v_per_person. A 1000-seat hall and a 100-seat hall then produce the same expected offspring at the same fill fraction. What the volume model adds over the present kernel is therefore fill fraction (a full house against a half-empty room), duration, and the cross-type contrast in m3 per person and air changes. It does not add "bigger room, more infections", and it should not: the `--densityExponent 0` run gave median occupancy at infection of 1,235 people because it implicitly gave a 1,000-person room the same size as a 5-person room.

This also means the denominator should be fixed capacity and not head count, otherwise the cancellation returns. The effect on homes needs care: at present a household of four with two people out has density = sum(I)/2. With fixed capacity it becomes sum(I)/4, and household transmission, which is 46% of infections in the flat baseline, changes. Recommendation: apply the volume model to non-residential types first, behind a flag, and leave residence on the head-count form until the household share has been compared.

## 3. How much dispersion could this plausibly buy? A toy calculation

The question that decides whether this is worth building is how heavy the tail of per-visit hazard has to be. Setup: each infector has a personal natural-history factor (CV2 0.19, matching the baseline early k of about 5.25), R = 1.4, and a number of visits over the infectious period whose expected offspring vary multiplicatively (mean preserved). Offspring are Poisson given the total. Numbers are from my simulation, 200,000 infectors per row.

| Scenario (30 visits per infector) | per-visit variation | resulting k |
|---|---|---:|
| no visit variation | none | 5.3 |
| lognormal, sigma 0.5 | CV 0.53 | 5.0 |
| lognormal, sigma 1.0 | CV 1.31 | 3.9 |
| lognormal, sigma 1.5 | CV 2.9 | 1.9 |
| lognormal, sigma 2.0 | CV 7.3 | 0.41 |
| lognormal, sigma 2.5 | CV 23 | 0.09 |
| 1% of visits at 10x | | 4.6 |
| 1% of visits at 50x | | 1.6 |
| 1% of visits at 200x | | 0.52 |
| 0.3% of visits at 1000x | | 0.13 |

| Scenario (only 5 relevant visits) | resulting k |
|---|---:|
| lognormal sigma 1.0 | 1.7 |
| lognormal sigma 1.5 | 0.48 |
| lognormal sigma 2.0 | 0.10 |

Reading this honestly:

- **Averaging over many visits washes variation out.** With 30 visits per infector, per-visit hazards need a coefficient of variation near 7 (sigma 2) to reach k of 0.4. A volume model driven by fill fraction, duration and ventilation will give per-visit variation of a factor of a few, not hundreds, so I expect k to move from about 5 to somewhere in the 2 to 4 range, not to 0.3. That is a guess about the size of the realistic variation, not a result.
- **A tail needs rare, extreme events.** The toy reaches published-range k only with about 1% of visits 200 times worse than average, or 0.3% 1000 times worse. Those are the Skagit-type events, and they are what the literature says drives the tail [R11, R2]. A fixed-volume model can represent a full, poorly ventilated, long, loud event (fill fraction near 1, ACH 0.3, three hours, singing), but the type-level table above only spreads effective volume by a factor of 2 to 4 within a type. The extreme end has to come from an explicit event module on top, which is the same conclusion as option D.
- **It matches the phase 2 finding in section 4.** The shape-0.3 multiplier survives the wave-2 hold-out and the venue-based variants do not. A volume model is unlikely to replace it, and is better seen as a way of reallocating where the heterogeneity sits, which matters for the intervention ranking (Set I against Set II).

So the realistic expectation is a modest k reduction plus a more defensible mechanism for venue-closure, occupancy-cap and ventilation interventions. It is not a route to k near 0.1 without events.

## 4. Educated-guess table

Effective removal per capacity-person is m3_per_person * (ACH + L), with L = 1 per hour. "Relative to home" divides by the residence value so that it can be slotted into the existing infection coefficient calibration. Low and high ACH bracket the plausible range; the spread column is the factor between high and low scenarios for effective volume, and is the within-type uncertainty to carry into a sensitivity run.

| ID | Type | m3 per person at capacity | ACH low / mid / high | Effective removal, mid (m3/h per person) | Relative to home | Basis |
|---|---|---:|---|---:|---:|---|
| 2 | Residence | 40 | 0.3 / 0.5 / 1.0 | 60 | 1.0 | shared living volume, windows closed in cold season |
| 3 | School stay site (non-classroom) | 15 | 1.0 / 2.0 / 4.0 | 45 | 0.75 | corridors, canteen, gym; mixed |
| 33 | Classroom | 6 | 0.7 / 1.5 / 4.0 | 15 | 0.25 | ~2 m2/pupil x ~3 m; natural ventilation dominates |
| 4 | Standard workplace | 30 | 1.0 / 2.0 / 4.0 | 90 | 1.5 | open office ~10 m2/person; factory/pharmacy larger |
| 13 | Non-standard-schedule workplace | 40 | 1.0 / 2.0 / 4.0 | 120 | 2.0 | night shifts, plants, services |
| 44 | University | 10 | 1.0 / 3.0 / 6.0 | 40 | 0.67 | lecture halls ~1.5-2 m2/seat, high ceilings |
| 5 | Restaurant / pub (20-100) | 6 | 1.0 / 3.0 / 6.0 | 24 | 0.4 | ~1.5-2 m2/seat x ~3.2 m |
| 6 | Cinema / theatre / concert (100-1000) | 6 | 3.0 / 6.0 / 10.0 | 42 | 0.7 | mechanical HVAC usual |
| 7 | Short-visit shop / front office | 25 | 2.0 / 4.0 / 8.0 | 125 | 2.08 | ~8 m2/person x ~3.5 m |
| 8 | Longer-visit shop / gym / playground | 30 | 1.0 / 3.0 / 8.0 | 120 | 2.0 | gym adds exertion emission, handled separately |
| 9 | Church (50-500) | 15 | 0.3 / 1.0 / 2.0 | 30 | 0.5 | high ceilings, poor ventilation, singing |
| 10 | Outdoor recreation | 500 | 10.0 / 20.0 / 50.0 | 10500 | 175.0 | treat as open air; cannot accumulate |
| 11 | Closed facility (prison) | 12 | 0.3 / 0.7 / 1.5 | 20 | 0.34 | shared cells, dayrooms |
| 12 | Hospital | 25 | 2.0 / 4.0 / 6.0 | 125 | 2.08 | wards 6 to 12 ACH for isolation, 2-4 elsewhere |
| 14 | Health centre | 12 | 1.0 / 3.0 / 6.0 | 48 | 0.8 | waiting rooms dense |
| 22 | Nursing home | 20 | 0.3 / 1.0 / 2.0 | 40 | 0.67 | shared rooms, poor ventilation |

Notes:

- **Type 1, public space, is omitted**: phase 1 found public locations carry infectiousness 0 in `locations0.json`, so the model cannot turn them on, and a volume would not change that.
- **Type 10, outdoor, is set so large that it cannot accumulate virus.** That reproduces the low base infectiousness (0.3) without a separate rule.
- **Gyms, singing and shouting change emission, not volume.** Emission per person varies by large factors with activity [R11 and the quanta literature]. I have not set emission multipliers here; they should be a separate per-location-type factor so that volume and emission can be varied independently.
- **Schools:** the model has both a school stay site (3) and a classroom (33); I have treated the stay site as the shared non-classroom spaces. Please correct if that is not how the generator uses them.
- **Hospital and nursing home** are placeholders for an aggregate; wards, isolation rooms and common rooms differ by a factor of ten in ACH.

## 5. Where to replace guesses with sources

Before any of these values are used in a reported result, they should be taken from published ventilation and occupancy standards and surveys instead of my estimates. Standards that are normally used for this purpose are the European ventilation standard for buildings (EN 16798-1) and the ASHRAE ventilation standard (62.1), plus measured air-change data for schools and homes in Hungary and neighbouring countries. I have not retrieved or verified any of these in this session. The quanta framework for the emission side is [R11] and Buonanno et al. 2020 (https://doi.org/10.1016/j.envint.2020.105794), whose abstract I read earlier.

## 6. Suggested test plan, in order

1. **Check `locations0.json`** for capacity, floor area or volume fields. If present, use them for Ncap_loc. If not, use assigned agent count and say so.
2. **Add the flag** `--volumeModel off|nonresidential|all`, with the table loaded from a JSON file so values can be varied without recompiling. Default off, so existing fits are untouched.
3. **Re-calibrate** the infection coefficient to early R of 1.39 as in phase 2, with the shed multiplier off, and measure early k, v10 (infections in the busiest 10% of venues) and the household share. This is the baseline comparison column.
4. **Sensitivity**: run low, mid and high ACH scenarios from the table, and a version with fill fraction fixed at 1, to separate what comes from ventilation contrast from what comes from fill-fraction variation.
5. **Events at fixed volume**: add the explicit event module on top and test the 0.3%-of-visits-at-1000x scenario from section 3. This is the only run that can tell us whether published-range k is reachable without the multiplier.
6. **Hold-out**: refit to the wave-1 hospital area and compare wave 2, as in phase 2 section 7. Only a configuration that survives this is a candidate for production.
7. **Interventions**: repeat the Set I / Set II lever experiment with a volume-model Set II, adding two lever types that are currently near no-ops: an occupancy cap, which now binds because fill fraction enters, and a ventilation improvement (scale ACH for a type).

## 7. What this changes in the earlier documents

- 03_pansim_options_table.md: option B (contact heterogeneity through schedules) was marked refuted on this kernel. A volume model is a distinct option B2 and should be added, with the modest-k expectation from section 3.
- 02_brief.md: the limitation should be worded as in the correction already made, that PanSim cannot make crowding amplify a single infector's offspring because of the head-count denominator, with a volume model as the proposed remedy.
