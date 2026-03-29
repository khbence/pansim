# Inputs And Configuration

## Overview

PanSim takes input from three layers:

1. command-line options
2. structured JSON input files
3. a few plain-text schedule files referenced by options

At startup, [DataProvider](/home/ireguly/pansim/include/IO/dataProviders/dataProvider.h) reads the configured files and optionally synthesizes random agents, locations, or states in [dataProvider.cpp](/home/ireguly/pansim/src/IO/dataProviders/dataProvider.cpp#L184).

## Startup Loading Order

The startup path is:

1. parse CLI options in [programOptions.cpp](/home/ireguly/pansim/src/runtime/programOptions.cpp#L7)
2. add simulation/policy/module-specific options from:
   - [simulation.h](/home/ireguly/pansim/include/simulation.h#L98)
   - [movementPolicies.h](/home/ireguly/pansim/include/policies/movementPolicies.h#L1395)
   - [testingPolicies.h](/home/ireguly/pansim/include/policies/testingPolicies.h#L318)
   - [infectionPolicies.h](/home/ireguly/pansim/include/policies/infectionPolicies.h#L89)
   - [closurePolicies.h](/home/ireguly/pansim/include/policies/closurePolicies.h#L73)
   - [agentsList.h](/home/ireguly/pansim/include/agentsList.h#L96)
   - [immunization.h](/home/ireguly/pansim/include/immunization.h#L91)
3. construct `DataProvider`
4. load parameters, progression config, closure rules, location types, agent types, locations, and agents
5. if requested, randomize locations, agents, or states

The loader behavior matters:

- `--numlocs != -1` generates random locations instead of reading the locations JSON
- `--numagents != -1` generates random agents instead of reading the agents JSON
- `--randomStates` keeps the loaded agents but replaces their disease states from `configRandom.json`

## Command-Line Options

## Core Files And Execution

| Option | Default | Meaning |
|---|---:|---|
| `-w`, `--weeks` | `12` | Simulation length in weeks. |
| `-t`, `--deltat` | `10` | Timestep length in minutes. |
| `-n`, `--numagents` | `-1` | If `-1`, read agents from file. Otherwise generate this many random agents. |
| `-N`, `--numlocs` | `-1` | If `-1`, read locations from file. Otherwise generate this many random locations. |
| `-P`, `--progression` | `inputConfigFiles/progressions/transition_config.json` | Progression directory/config file. |
| `-a`, `--agents` | `inputRealExample/agents.json` | Agents JSON file. |
| `-A`, `--agentTypes` | `inputConfigFiles/agentTypes.json` | Agent-type/schedule JSON file. |
| `-l`, `--locations` | `inputRealExample/locations.json` | Locations JSON file. |
| `-L`, `--locationTypes` | `inputConfigFiles/locationTypes.json` | Location-types JSON file. |
| `-p`, `--parameters` | `inputConfigFiles/parameters.json` | General parameter JSON file. |
| `-c`, `--configRandom` | `inputConfigFiles/configRandom.json` | Random generation and state-randomization config. |
| `--closures` | `inputConfigFiles/closureRules.json` | Closure/intervention rule JSON file. |
| `-r`, `--randomStates` | disabled | Replace loaded agent states from `configRandom.stateDistribution`. |
| `--outAgentStat` | `""` | If non-empty, write per-agent stats JSON to this file. |
| `--seed` | unset | Base RNG seed. Required for repeatable deterministic runs. |
| `--threads` | unset | Override OpenMP thread count before setup. |
| `--diags` | `0` | Diagnostic verbosity level. |
| `-h`, `--help` | n/a | Print usage. |
| `--version` | n/a | Print build version. |

## Simulation-Wide Epidemiology And Calendar

| Option | Default | Meaning |
|---|---:|---|
| `--otherDisease` | `1` | Enable non-COVID hospitalization/sudden-death logic. |
| `--infectiousnessMultiplier` | `1.03,1.79,2.45,2.7,3.6` | Per-strain infectiousness multipliers. First value is baseline strain. |
| `--diseaseProgressionScaling` | `0.9,1.05,0.9,0.8,0.8` | Per-strain progression scaling. |
| `--diseaseProgressionDeathScaling` | `1.0,1.03,1.3,0.6,0.6` | Per-strain death scaling. |
| `--startDay` | `2` | Day-of-week index for simulation start. Monday is `0`. |
| `--startDate` | `267` | Day-of-year index for simulation start. January 1 is `0`. |
| `--totalHospitalizations` | `inputConfigFiles/dailyHospitalizationTargets.txt` | Plain-text daily non-COVID hospitalization targets. |
| `--dueWithCOVID` | `0` | Whether total hospitalization target includes unrelated + due-to-COVID (`0`) or unrelated + with-COVID (`1`). |

## Movement And Quarantine

Defined by the configured movement policy in [movementPolicies.h](/home/ireguly/pansim/include/policies/movementPolicies.h#L1395).

| Option | Default | Meaning |
|---|---:|---|
| `--trace` | `max unsigned` | Track a single agent id for diagnostic movement/event prints. |
| `--quarantinePolicy` | `3` | `0` none, `1` agent only, `2` agent + household, `3` + classroom/work, `4` + school. |
| `--quarantineLength` | `10` | Quarantine length in days. |
| `--dumpLocationAgentList` | `""` | If non-empty, dump per-location agent lists each iteration. |
| `--dumpLoctypeStat` | `""` | If non-empty, append per-location-type occupancy snapshots. |

## Testing

Defined by [testingPolicies.h](/home/ireguly/pansim/include/policies/testingPolicies.h#L318).

| Option | Default | Meaning |
|---|---:|---|
| `--testingProbabilities` | `0.00005,0.01,0.0005,0.0005,0.005,0.05` | Random/home/work/school/hospital/nursery-home testing probabilities. |
| `--testingRepeatDelay` | `5` | Minimum days between tests. |
| `--testingMethod` | `PCR` | `PCR` or `antigen`. Uses accuracy values from progression state metadata. |

## Infection

Defined by [infectionPolicies.h](/home/ireguly/pansim/include/policies/infectionPolicies.h#L89).

| Option | Default | Meaning |
|---|---:|---|
| `-k`, `--infectionCoefficient` | `0.000347` | Base infectiousness coefficient. |
| `--dumpLocationInfections` | `0` | Dump per-location infection statistics every N timesteps. |
| `--dumpLocationInfectiousList` | `""` | If non-empty, dump per-location infectious lists. |
| `--d_offset` | `-5` | Seasonality day offset. |
| `--d_peak_offset` | `0` | Seasonality peak-day offset. |
| `--trunc` | `0.5` | Omicron seasonality truncation parameter. |
| `--c0` | `3.08` | Seasonality coefficient. |

## Closures And Interventions

Defined by [closurePolicies.h](/home/ireguly/pansim/include/policies/closurePolicies.h#L73).

| Option | Default | Meaning |
|---|---:|---|
| `--enableClosures` | `1` | Enable or disable closure rules from `closureRules.json`. |

## Agents

Defined by [agentsList.h](/home/ireguly/pansim/include/agentsList.h#L96).

| Option | Default | Meaning |
|---|---:|---|
| `--disableTourists` | `1` | Skip agents with `typeID == 9` during initialization. |

## Immunization And Boosters

Defined by [immunization.h](/home/ireguly/pansim/include/immunization.h#L91).

| Option | Default | Meaning |
|---|---:|---|
| `--immunizationStart` | `96` | Day index when vaccination starts. |
| `--boosterStart` | `315` | Comma-separated list of day indices for booster rounds. |
| `--immunizationsPerWeek` | `inputConfigFiles/vaccPerWeek.txt` | Either a number or a file with comma-separated weekly values. |
| `--boostersPerWeek` | `inputConfigFiles/boosterPerWeek.txt` | Either a number or a file describing booster rounds. |
| `--immunizationOrder` | `1,2,3,4,5,6,0,0,7,8` | Priority ordering of 10 vaccination categories. `0` disables a category. |
| `--vaccinationGroupLevel` | `0.9,0.85,0.9,0.82,0.8,0.75,0.8,0.67,0.4,0.2` | Acceptance/coverage level for each vaccination category. |
| `--protectionInfection` | see code | Infection protection values for prior infection / doses / combinations. |
| `--protectionInfectionWaning` | see code | Weekly waning rates corresponding to `protectionInfection`. |
| `--protectionSymptomatic` | see code | Symptomatic-disease protection values. |
| `--protectionSymptomaticWaning` | see code | Weekly waning rates for symptomatic protection. |
| `--protectionHospitalization` | see code | Hospitalization protection values. |
| `--protectionHospitalizationWaning` | see code | Weekly waning rates for hospitalization protection. |
| `--variantSimilarity` | `0,0,0,1,1,1,1` | Integer similarity class for each variant. |
| `--variantSimilarMultiplier` | `0.0,0.0,0.0,0.0,0.0,0.0,0.0` | Modifier for similar-strain immune escape. |

For the long default vectors, the authoritative defaults are in [include/immunization.h](/home/ireguly/pansim/include/immunization.h#L91).

## Input File Set

The default configuration expects these files:

- `parameters.json`
- `progressions/transition_config.json`
- one progression matrix JSON per `(age range, precondition)` referenced from the progression config
- `locationTypes.json`
- `agentTypes.json`
- `locations.json`
- `agents.json`
- `closureRules.json`
- optionally `configRandom.json`
- optionally plain-text hospitalization / vaccination / booster schedule files

## JSON Input Structures

The schemas below are the actual decoded structures from `include/IO/inputFormats/*.h`.

## 1. Parameters JSON

Header: [parametersFormat.h](/home/ireguly/pansim/include/IO/inputFormats/parametersFormat.h#L6)

```json
{
  "sex": [
    { "name": "M", "symptoms": 1.0 }
  ],
  "age": [
    { "from": 0, "to": 10, "symptoms": 1.0, "transmission": 1.0 }
  ],
  "preCondition": [
    { "ID": "none", "condition": "None", "symptoms": 1.0 }
  ]
}
```

Notes:

- `preCondition[].ID` is used later by progression configuration and by agents.
- `age` ranges drive metadata-derived symptom and transmission scaling.

## 2. Progression Directory JSON

Header: [progressionConfigFormat.h](/home/ireguly/pansim/include/IO/inputFormats/progressionConfigFormat.h#L6)

```json
{
  "stateInformation": {
    "stateNames": ["S", "E", "I1"],
    "firstInfectedState": "E",
    "nonCOVIDDeadState": "D2",
    "susceptibleStates": ["S"],
    "infectedStates": ["E", "I1"]
  },
  "transitionMatrices": [
    {
      "fileName": "matrix_age0_9_none.json",
      "age": [0, 10],
      "preCond": "none"
    }
  ],
  "states": [
    {
      "stateName": "I1",
      "WB": "I",
      "infectious": 1.0,
      "accuracyPCR": 0.95,
      "accuracyAntigen": 0.8
    }
  ]
}
```

Notes:

- `transitionMatrices[].fileName` is resolved relative to the progression config file directory in [dataProvider.cpp](/home/ireguly/pansim/src/IO/dataProviders/dataProvider.cpp#L11).
- Each `(age range, preCond)` entry must map to a matrix file.

## 3. Progression Matrix JSON

Header: [progressionMatrixFormat.h](/home/ireguly/pansim/include/IO/inputFormats/progressionMatrixFormat.h#L6)

```json
{
  "states": [
    {
      "stateName": "E",
      "avgLength": [2.0],
      "maxlength": [5.0],
      "progressions": [
        { "name": "I1", "chance": 1.0, "isBadProgression": false }
      ]
    }
  ]
}
```

Notes:

- `avgLength` and `maxlength` are vectors, not scalars.
- `progressions[].name` must refer to known state names from the progression directory.

## 4. Location Types JSON

Header: [locationTypesFormat.h](/home/ireguly/pansim/include/IO/inputFormats/locationTypesFormat.h#L6)

```json
{
  "publicSpace": 1,
  "home": 2,
  "hospital": 12,
  "doctor": 14,
  "school": 3,
  "classroom": 33,
  "work": 4,
  "nurseryhome": 22,
  "types": [
    { "ID": 2, "name": "home" }
  ]
}
```

Notes:

- The named ids above are used directly by movement, testing, and immunization logic.
- `types[]` is also used to build the runtime id-to-name table.

## 5. Locations JSON

Header: [locationsFormat.h](/home/ireguly/pansim/include/IO/inputFormats/locationsFormat.h#L6)

```json
{
  "places": [
    {
      "ID": "home_0001",
      "type": 2,
      "essential": 0,
      "coordinates": [0.0, 0.0],
      "infectious": 1.0,
      "area": 100,
      "state": "ON",
      "capacity": 4,
      "ageInter": [0, 100]
    }
  ]
}
```

Notes:

- `state` must be `ON`/`OPEN` or `OFF`/`CLOSED`; validation happens in [locationListRuntime.cpp](/home/ireguly/pansim/src/runtime/locationListRuntime.cpp#L81).
- Classroom ids are expected to follow `class_school` naming so they can be linked back to schools in [locationListRuntime.cpp](/home/ireguly/pansim/src/runtime/locationListRuntime.cpp#L101).

## 6. Agent Types JSON

Header: [agentTypesFormat.h](/home/ireguly/pansim/include/IO/inputFormats/agentTypesFormat.h#L6)

```json
{
  "types": [
    {
      "name": "worker",
      "ID": 4,
      "schedulesUnique": [
        {
          "WB": "W",
          "dayType": "weekday",
          "schedule": [
            {
              "type": 4,
              "chance": 1.0,
              "start": 8.0,
              "end": 16.0,
              "duration": 8.0
            }
          ]
        }
      ]
    }
  ]
}
```

Notes:

- `WB` and `dayType` are interpreted by `states::parseWBState` and `Timehandler::parseDays`.
- Every schedule event `type` contributes to the required location-type set for agents of that type in [dataProvider.cpp](/home/ireguly/pansim/src/IO/dataProviders/dataProvider.cpp#L57).

## 7. Agents JSON

Header: [agentsFormat.h](/home/ireguly/pansim/include/IO/inputFormats/agentsFormat.h#L6)

```json
{
  "people": [
    {
      "age": 42,
      "sex": "M",
      "preCond": "none",
      "state": "S",
      "typeID": 4,
      "locations": [
        { "typeID": 2, "locID": "home_0001" },
        { "typeID": 4, "locID": "work_0102" }
      ]
    }
  ]
}
```

Notes:

- `typeID` must exist in `agentTypes.json`.
- every `locations[].locID` must exist in `locations.json`
- every agent must provide all required location types implied by its `typeID`, or initialization emits a missing-location-type warning in [agentsListRuntime.cpp](/home/ireguly/pansim/src/runtime/agentsListRuntime.cpp#L145)
- the current formal decoded schema does not declare a JSON `diagnosed` field even though runtime agent initialization uses `person.diagnosed`; if you rely on diagnosed-at-start behavior, verify against your current input files and runtime expectations

## 8. Closure Rules JSON

Header: [closuresFormat.h](/home/ireguly/pansim/include/IO/inputFormats/closuresFormat.h#L6)

```json
{
  "rules": [
    {
      "name": "Masks",
      "conditionType": "afterDays",
      "threshold": 0.0,
      "threshold2": 0.0,
      "parameter": "0.8",
      "closeAfter": -1,
      "openAfter": 30,
      "locationTypesToClose": [3, 4]
    }
  ]
}
```

Notes:

- the exact interpretation of `conditionType`, `parameter`, thresholds, and delays is implemented in [closurePolicies.h](/home/ireguly/pansim/include/policies/closurePolicies.h)
- rule names are not just labels; some names trigger special behaviors such as masks, curfew, holiday mode, quarantine policy, testing probability, and lockdown controls

## 9. Random Config JSON

Header: [configRandomFormat.h](/home/ireguly/pansim/include/IO/inputFormats/configRandomFormat.h#L6)

Used only when:

- random locations are requested
- random agents are requested
- `--randomStates` is used

```json
{
  "irregularLocationChance": {
    "generalChance": 0.1,
    "detailsOfChances": [
      {
        "value": "4",
        "chanceForType": 0.2,
        "chanceFromAllIrregular": 0.3,
        "switchedToWhat": [
          { "value": "7", "chance": 1.0 }
        ]
      }
    ]
  },
  "locationTypeDistribution": [
    { "value": "2", "chance": 0.5 }
  ],
  "preCondDistribution": [
    { "value": "none", "chance": 0.8 }
  ],
  "stateDistribution": [
    {
      "ageStart": 0,
      "ageEnd": 100,
      "distribution": [
        { "value": "S", "chance": 0.99, "diagnosedChance": 0.0 }
      ]
    }
  ],
  "agentTypeDistribution": [
    { "value": "4", "chance": 1.0 }
  ]
}
```

Notes:

- `locationTypeDistribution`, `preCondDistribution`, `stateDistribution`, and `agentTypeDistribution` drive random generation in [dataProvider.cpp](/home/ireguly/pansim/src/IO/dataProviders/dataProvider.cpp#L97)
- classroom random generation has special handling for school pairing

## Plain-Text Schedule Files

## Daily Hospitalization Targets

Used by `--totalHospitalizations`.

Format:

- one line per day
- empty lines and lines beginning with `#` are ignored
- each data line is parsed as comma-separated integers

Loader: [simulationOrchestration.cpp](/home/ireguly/pansim/src/runtime/simulationOrchestration.cpp#L410)

## Vaccinations Per Week

Used by `--immunizationsPerWeek` when the value is not a plain integer.

Format:

- file contents are read as a single comma-separated float list

Loader: [immunizationRuntime.cpp](/home/ireguly/pansim/src/runtime/immunizationRuntime.cpp#L8)

## Boosters Per Week

Used by `--boostersPerWeek` when the value is not a plain integer.

Format:

- lines beginning with `#` or empty lines are ignored
- expected as pairs of lines per booster round:
  - line 1: comma-separated weekly booster counts
  - line 2: comma-separated age-group percentages

The number of rounds must match `--boosterStart`.

Loader: [immunizationRuntime.cpp](/home/ireguly/pansim/src/runtime/immunizationRuntime.cpp#L23)

## File Relationships And Consistency Rules

These are the main cross-file constraints:

- every `agents.typeID` must exist in `agentTypes.types[].ID`
- every `agents.locations[].locID` must exist in `locations.places[].ID`
- every `agents.locations[].typeID` should satisfy the required location types implied by the agent type schedule
- every progression config entry `(age range, preCond)` should match an actual precondition id from `parameters.preCondition[].ID`
- every progression config matrix entry should point to an existing matrix file
- school/classroom location ids must be compatible with the classroom-to-school pairing logic

## Deterministic And Randomized Setups

For deterministic CPU setup:

```bash
./panSim --seed 1234 --threads 1 ...
```

For generated/randomized populations:

```bash
./panSim -n 10000 -N 2000 -c inputConfigFiles/configRandom.json
```

For file-based agents with randomized disease states:

```bash
./panSim -a inputRealExample/agents.json -r -c inputConfigFiles/configRandom.json
```

## Recommended Documentation Maintenance

When adding a new option or input field:

1. add the parsing/option definition in code
2. update this document
3. if it changes deterministic CPU behavior, update the reference tests and note the new expectation

This file should be treated as the user-facing map of the simulator’s configuration surface, while [ARCHITECTURE.md](/home/ireguly/pansim/ARCHITECTURE.md#L1) describes the internal design.
