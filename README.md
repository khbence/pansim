# Pansim
Pansim is a pandemic simulation framework of PPCU university. It uses agent-based modelling to simulate the spread of the disease. Every agent represents a human, which has movement depending on its schedule and locations. Infectiousness is spreading between agents in the same locations in a stochastic way. The main goal and the default parameters are specialized for the current COVID-19 pandemic, but with correct parameters it can be applied for many epidemic situations.

## Technical parameters
The simulation can be run on CPU only using OpenMP to accelerate or it can also use CUDA capable GPU. On a larger 16 GB RAM GPU we can simulate up to 50 million agents, but even on a smaller regular GPU a few million agent can be simulated safely.

## Usage

### **Input options**
Details about the format and contents in input json files are discussed [here](inputFiles/README.md)
|short|long|details
|--- | --- | ---
|  -w| --weeks                   | Length of simulation in weeks (default: 12)
|  -t| --deltat                  | Length of timestep in minutes (default: 10)
|  -n| --numagents               | Number of agents (default: -1)
|  -N| --numlocs                 | Number of dummy locations (default: -1)
|  -P| --progression             | Path to the config file for the progression matrices. (default: ../inputFiles/progressions/transition_config.json)
|  -a| --agents                  | Agents file, for all human being in the experiment. (default: ../inputFiles/agents.json)
|  -A| --agentTypes              | List and schedule of all type fo agents. (default: ../inputFiles/agentTypes.json)
|  -l| --locations               | List of all locations in the simulation. (default: ../inputFiles/locations.json)
|  -L| --locationTypes           | List of all type of locations (default: ../inputFiles/locationTypes.json)
|  -p| --parameters              | List of all general parameters for the simulation except the progression data. (default: ../inputFiles/parameters.json)
|  -c| --configRandom            | Config file for random initialization. (default: ../inputFiles/configRandom.json)
|    | --closures                | List of closure rules. (default: ../inputFiles/closureRules.json)
|  -r| --randomStates            | Change the states from the agents file with the configRandom file's stateDistribution.
|    | --outAgentStat            | name of the agent stat output file, if not set there will be no print (default: "")
|    | --diags                   | level of diagnositcs to print (default: 0)
|    | --otherDisease            | Enable (1) or disable (0) non-COVID related hospitalization and sudden death  (default: 1)
|    | --mutationMultiplier      | infectiousness multiplier for mutated virus (default: 1.0)
|  -k| --infectionCoefficient    | Infection: >0 :infectiousness coefficient (default: 0.000374395)
|    | --dumpLocationInfections  | Dump per-location statistics every N timestep  (default: 0)
|    | --dumpLocationInfectiousList | Dump per-location list of infectious people (default: "")
|    | --trace                   | Trace movements of agent (default: 4294967295)
|    | --quarantinePolicy        | Quarantine policy: 0 - None, 1 - Agent only, 2 - Agent and household, 3 - + classroom/work, 4 - + school (default: 3)
|    | --quarantineLength        | Length of quarantine in days (default: 10)
|    | --testingProbabilities    | Testing probabilities for random, if someone else was diagnosed at home/work/school, and random for hospital workers: comma-delimited string random,home,work,school,hospital,nurseryHome (default: 0.0001,0.02,0.001,0.001,0.01,0.1)
|    | --testingRepeatDelay      | Minimum number of days between taking tests (default: 5)
|    | --testingMethod           | default method for testing. Can be PCR (default) on antigen. Accuracies are provided in progression json input (default: PCR)
|    | --enableClosures          | Enable(1)/disable(0) closure rules defined in closureRules.json (default: 1)
|    | --disableTourists         | enable or disable tourists (default: 1)
|    | --immunizationStart       | number of days into simulation when immunization starts (default: 0)
|    | --immunizationsPerDay     | number of immunizations per day (default: 0)
|    | --immunizationOrder       | Order of immunization (starting at 1, 0 to skip) for agents in different categories health workers, nursery home worker/resident, 60+, 18-60 with underlying condition, essential worker, 18+ (default: 1,2,3,4,5,6)
|  -h| --help                    | Print usage
|    | --version| 

### **Using Docker image**
There are many ways to run this simulation. There are ready to use docker images at [khbence/covid_ppcu](https://hub.docker.com/r/khbence/covid_ppcu).

To run on CPU and accelerate it using OpenMP use the following command:
> docker run khbence/covid_ppcu:cpu *args...*

To utilise CUDA capable GPU:
> docker run khbence/covid_ppcu:gpu *args...*

The images contain the inputFiles directory with the default config files, but be aware that a docker container not going to see the directories of the host system by default (for example for input and output). For that you need to [mount](https://docs.docker.com/storage/bind-mounts/) the needed directories.

If you want to create your own image from this project, you can use the make commands from the [Makefile](Makefile). Use dbuildCPU/dbuildGPU to build and then use similar docker run commands with the created images names, which is the same as the ones from the Docker Hub links above.
> make dbuildCPU

*or*

> make dbuildGPU

### **Using local source build**
Use the [Makefile](Makefile) to build it for GPU or CPU, run the buildCPU or buildGPU. The compiler need to be C++17 with OpenMP capabilities and the CUDA Toolkit for CUDA-GPU builds. 
> make buildCPU

*or*

> make buildGPU

## Input files


# License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg