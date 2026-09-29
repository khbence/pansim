# Architecture

## Purpose

PanSim is an agent-based epidemic simulator with:

- a native CLI executable: `panSim`
- optional MATLAB and Python bindings built from the same core startup path
- CPU execution through OpenMP/Thrust
- GPU execution through CUDA/Thrust

The codebase is organized around a simulation template that composes domain state and policy implementations, with a runtime bootstrap layer that hides CLI parsing and engine setup from the frontends.

## High-Level Structure

The project is split across a few major areas:

- `include/`
  Public and internal headers for simulation state, policies, tools, IO, and the runtime bootstrap API.
- `src/`
  Non-header implementation units for runtime bootstrap, tools, IO, agent data types, and other host-side logic.
- `matlab/`
  MATLAB-facing integration, studies, and the shared mex entrypoint.
- `test/`
  Lightweight regression and determinism tests.
- `scripts/`
  Manual and CI-friendly helper scripts.

## Build Architecture

The top-level build is defined in [CMakeLists.txt](/home/ireguly/pansim/CMakeLists.txt). Key decisions:

- `USE_GPU=ON/OFF` selects CUDA or CPU mode.
- `ARCHITECTURE` controls CUDA architecture selection.
- `MovementPolicy`, `TestingPolicy`, and `ClosurePolicy` are chosen at CMake configure time and written into generated config in `cmake/config/out/configTypes.h`.
- `ENABLE_MATLAB` and `ENABLE_PYTHON` enable the binding targets.

The executable and bindings are assembled from subsystem CMake fragments included by [src/CMakeLists.txt](/home/ireguly/pansim/src/CMakeLists.txt).

### Runtime Object Target

The refactored startup and host-only orchestration path is built as the `panSimRuntime` object target in [src/runtime/CMakeLists.txt](/home/ireguly/pansim/src/runtime/CMakeLists.txt).

That target now compiles separate implementation units for:

- program option construction
- simulation bootstrap
- simulation orchestration
- agent list host initialization/reporting
- location list host initialization/reporting
- immunization host parsing/setup

This separation matters most for CUDA builds, where avoiding a single giant translation unit reduces rebuild cost and device-link churn.

## Entry Points

### CLI

The CLI executable entrypoint is [src/main.cpp](/home/ireguly/pansim/src/main.cpp). It does three things:

1. calls `runtime::bootstrapSimulation(...)`
2. exits early for `--help` or `--version`
3. runs the simulation engine

The CLI is intentionally thin. Most startup behavior now lives behind the runtime API in [include/runtime/simulationRuntime.h](/home/ireguly/pansim/include/runtime/simulationRuntime.h).

### MATLAB and Python

MATLAB and Python bindings reuse the same runtime bootstrap path through `matlab/mexPanSim.cpp`, rather than maintaining separate startup logic.

## Main Runtime Layers

### 1. Runtime Bootstrap Layer

Files:

- [include/runtime/simulationRuntime.h](/home/ireguly/pansim/include/runtime/simulationRuntime.h)
- [include/runtime/programOptions.h](/home/ireguly/pansim/include/runtime/programOptions.h)
- [src/runtime/programOptions.cpp](/home/ireguly/pansim/src/runtime/programOptions.cpp)
- [src/runtime/simulationRuntime.cpp](/home/ireguly/pansim/src/runtime/simulationRuntime.cpp)

Responsibilities:

- define top-level CLI options
- parse CLI arguments
- handle `--help` and `--version`
- configure threads and RNG seeding
- construct the configured simulation type
- expose a small `SimulationEngine` wrapper to frontends

This layer is the only place that should know about frontend bootstrap mechanics.

### 2. Simulation Orchestration Layer

Files:

- [include/simulation.h](/home/ireguly/pansim/include/simulation.h)
- [src/runtime/simulationOrchestration.cpp](/home/ireguly/pansim/src/runtime/simulationOrchestration.cpp)

`Simulation<...>` is still the central simulation type, parameterized by:

- position type
- location type
- progression state type
- agent metadata type
- movement policy
- infection policy
- testing policy
- closure policy

The header now mostly contains:

- type aliases
- owned pointers/references to major subsystems
- small control helpers
- declarations for host-side orchestration methods

The heavy host-side implementation now lives in `simulationOrchestration.cpp`, including:

- constructor/setup
- non-COVID disease progression logic
- agent state updates
- statistics printing
- simulation loop execution
- finalization

### 3. State and Domain Modules

#### Agent State

Files:

- [include/agentsList.h](/home/ireguly/pansim/include/agentsList.h)
- [src/runtime/agentsListRuntime.cpp](/home/ireguly/pansim/src/runtime/agentsListRuntime.cpp)

`AgentList` owns the main per-agent state vectors:

- progression states
- metadata
- stats
- diagnosis flags
- quarantine flags
- current location
- possible locations and their offsets

Host-side responsibilities now split cleanly into:

- `addProgramParameters`
- `initializeArgs`
- `initAgentTypes`
- `initAgents`
- `printAgentStatJSON`

The initialization and reporting implementation is out-of-line; hot-path access remains in the header where needed.

#### Location State

Files:

- [include/locationList.h](/home/ireguly/pansim/include/locationList.h)
- [src/runtime/locationListRuntime.cpp](/home/ireguly/pansim/src/runtime/locationListRuntime.cpp)

`LocationsList` owns:

- location metadata
- location status vectors
- school/classroom indexing
- per-location agent index structures

The host-only setup flow is now separated into:

- `initLocationTypes`
- `initializeArgs`
- `initLocations`
- `initialize`
- `refreshAndGetStatistic`

The infection kernel path still stays in the header because it is tightly coupled to the simulation hot path.

#### Immunization

Files:

- [include/immunization.h](/home/ireguly/pansim/include/immunization.h)
- [src/runtime/immunizationRuntime.cpp](/home/ireguly/pansim/src/runtime/immunizationRuntime.cpp)

`Immunization` mixes:

- immunization schedule parsing
- booster schedule parsing
- group/category assignment
- immunity waning and update logic

The config/setup side has been made more structured:

- `initializeArgs`
- `initCategories`
- `initAgeGroups`

Those are now implemented in `immunizationRuntime.cpp` with small local parsing helpers, while the runtime immunity update loop remains in the header.

### 4. Policy Layer

Files:

- [include/policies/movementPolicies.h](/home/ireguly/pansim/include/policies/movementPolicies.h)
- [include/policies/infectionPolicies.h](/home/ireguly/pansim/include/policies/infectionPolicies.h)
- [include/policies/testingPolicies.h](/home/ireguly/pansim/include/policies/testingPolicies.h)
- [include/policies/closurePolicies.h](/home/ireguly/pansim/include/policies/closurePolicies.h)

Policies define runtime behavior chosen at configure time. Today they are still primarily header-driven.

Their roles are:

- movement: schedules, travel decisions, school/work behavior, quarantine-related relocation
- infection: infectiousness accumulation and infection application
- testing: diagnosis, testing probabilities, quarantine triggering
- closure: rules driven by simulation statistics and intervention conditions

These policies still contain the largest concentration of header-defined logic. They are the main remaining area for further decomposition.

## Generated Configuration and Concrete Simulation Type

The configured simulation type is generated at CMake time into `cmake/config/out/configTypes.h`.

That generated file chooses the concrete aliases used by the runtime layer, including the final `config::Simulation_t`.

This means:

- the runtime bootstrap is non-template at the API level
- the underlying simulation remains compile-time composed
- changing a policy selection still requires reconfiguration/rebuild

## Data Flow

### Startup

1. `main` calls `runtime::bootstrapSimulation`
2. CLI options are defined and parsed
3. thread count and RNG are configured
4. `config::Simulation_t` is constructed
5. the simulation constructor loads config and input data via `DataProvider`
6. agents, locations, policies, and immunization are initialized

### Simulation Loop

At a high level, the loop in `Simulation` coordinates:

- movement planning and movement
- infection accumulation and infection application
- testing/quarantine updates
- closures/interventions
- disease progression updates
- optional non-COVID hospitalization/death logic
- immunization updates
- per-day statistics refresh and output

### Output

Primary outputs include:

- stdout tabular daily state summaries
- optional per-agent JSON output via `--outAgentStat`
- optional diagnostic prints depending on `--diags`

## Determinism Model

CPU determinism for single-thread runs is now explicitly supported.

Relevant files:

- [src/tools/randomGenerator.cpp](/home/ireguly/pansim/src/tools/randomGenerator.cpp)
- [include/tools/randomGenerator.h](/home/ireguly/pansim/include/tools/randomGenerator.h)
- [test/run_determinism_reference.sh](/home/ireguly/pansim/test/run_determinism_reference.sh)
- [scripts/test_determinism_cpu.sh](/home/ireguly/pansim/scripts/test_determinism_cpu.sh)

Important constraints:

- deterministic CPU runs require a fixed `--seed`
- deterministic CPU runs require `--threads 1`
- GPU atomics are explicitly documented as nondeterministic unless configured off

## Current Architectural Direction

The codebase is in transition from a mostly-header implementation toward a more modular split:

- thin runtime/bootstrap API
- host-only setup and orchestration in `.cpp` files
- subsystem-local initialization kept close to the subsystem
- reduced compile fanout from `simulation.h`, `locationList.h`, and `agentsList.h`
- runtime sources compiled as separate units again, including under CUDA

The project is not yet fully runtime-composed. The core simulation is still heavily template- and policy-driven, especially in the policy layer.

## Main Remaining Technical Debt

### Header-Heavy Policies

The policy headers still dominate compile time and architectural coupling. In particular:

- movement policy host setup and CUDA helpers
- testing policy setup and execution
- infection policy setup and aggregation helpers
- closure rule parsing and rule execution

### Singleton-Style Access

`AgentList` and `LocationsList` still expose `getInstance()` and are not yet explicit owned services. This keeps access simple but couples subsystems globally.

### Mixed Host and Device Concerns

Several modules still interleave:

- CLI/config parsing
- host-only initialization
- device-compatible kernels and lambdas

This is functional, but it increases cognitive load and slows compilation.

## Recommended Next Refactor Steps

1. Continue pulling host-only `initializeArgs` and `init(...)` logic out of policy headers.
2. Separate policy configuration parsing from policy execution where possible.
3. Introduce clearer subsystem boundaries for owned runtime services instead of relying on singleton access.
4. Reduce broad includes in central headers and replace them with forward declarations where feasible.
5. Keep GPU device helpers header-local only when required for kernels or inlining; move the rest out-of-line.

## Practical Reading Order

For someone new to the codebase, the best order is:

1. [CMakeLists.txt](/home/ireguly/pansim/CMakeLists.txt)
2. [src/main.cpp](/home/ireguly/pansim/src/main.cpp)
3. [include/runtime/simulationRuntime.h](/home/ireguly/pansim/include/runtime/simulationRuntime.h)
4. [src/runtime/simulationRuntime.cpp](/home/ireguly/pansim/src/runtime/simulationRuntime.cpp)
5. [include/simulation.h](/home/ireguly/pansim/include/simulation.h)
6. [src/runtime/simulationOrchestration.cpp](/home/ireguly/pansim/src/runtime/simulationOrchestration.cpp)
7. state modules: agent, location, immunization
8. policy headers

That order reflects the current execution path and the refactored architectural seams.
