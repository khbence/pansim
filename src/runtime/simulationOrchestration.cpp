#include "simulation.h"
#include "configTypes.h"

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
void Simulation<PositionType,
    TypeOfLocation,
    PPState,
    AgentMeta,
    MovementPolicy,
    InfectionPolicy,
    TestingPolicy,
    ClosurePolicy>::otherDisease(Timehandler& simTime, unsigned timeStep) {
    PROFILE_FUNCTION();
    auto& ppstates = agents->PPValues;
    auto& agentStats = agents->agentStats;
    auto& agentMeta = agents->agentMetaData;
    unsigned timestamp = simTime.getTimestamp();

    unsigned inHospitalWithCovid;
    if (dueWithCOVID == 0) {
        inHospitalWithCovid = thrust::count_if(ppstates.begin(), ppstates.end(), [] HD(PPState state) {
            return state.getStateIdx() >= 6 && state.getStateIdx() <= 8;
        });
    } else {
        unsigned hospitalTypeLocal = hospitalType;
        inHospitalWithCovid = thrust::count_if(
            thrust::make_zip_iterator(thrust::make_tuple(
                ppstates.begin(),
                thrust::make_permutation_iterator(locs->locType.begin(), agents->location.begin()),
                agentStats.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                ppstates.end(),
                thrust::make_permutation_iterator(locs->locType.begin(), agents->location.end()),
                agentStats.end())),
            [hospitalTypeLocal, timestamp] HD(thrust::tuple<PPState, unsigned, AgentStats> tup) {
                return thrust::get<0>(tup).isInfected() &&
                    thrust::get<1>(tup) == hospitalTypeLocal &&
                    !(thrust::get<0>(tup).getWBState() < states::WBStates::S &&
                        timestamp > 0 &&
                        thrust::get<2>(tup).hospitalizedUntilTimestamp == timestamp);
            });
    }

    float probabilityMul = 0.0f;
    unsigned simDay = simTime.getTimestamp() / simTime.getStepsPerDay();
    if (simDay < totalHospitalizations.size()) {
        unsigned targetHospitalized = std::max(
            0.0,
            double(totalHospitalizations[simDay]) - double(inHospitalWithCovid) * (9730000.0 / 179500.0));
        probabilityMul = float(targetHospitalized) / 48000.0;
    }

    unsigned tracked = locs->tracked;
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(
            ppstates.begin(), agentMeta.begin(), agentStats.begin(), thrust::make_counting_iterator<unsigned>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(
            ppstates.end(),
            agentMeta.end(),
            agentStats.end(),
            thrust::make_counting_iterator<unsigned>(0) + ppstates.size())),
        [timestamp, tracked, timeStep, probabilityMul] HD(thrust::tuple<PPState&, AgentMeta&, AgentStats&, unsigned> tup) {
            auto& ppstate = thrust::get<0>(tup);
            auto& meta = thrust::get<1>(tup);
            auto& agentStat = thrust::get<2>(tup);
            unsigned agentID = thrust::get<3>(tup);

            if (ppstate.getWBState() == states::WBStates::D) {
                return;
            }

            double randomHospProbs[] = {0.00017785,
                0.000118567,
                0.000614453,
                0.003174864,
                0.001016573,
                0.0011019,
                0.000926381,
                9.52062E-05,
                6.34708E-05,
                0.000328928,
                0.000986754,
                0.00044981,
                0.000623826,
                0.000855563,
                0.006333678,
                0.004222452,
                0.008444904,
                0.030227275,
                0.036202732,
                0.03924143,
                0.032990764,
                0.003390532,
                0.002260355,
                0.00452071,
                0.009394699,
                0.016018856,
                0.022216016,
                0.030468759,
                0.000364207,
                0.000242805,
                0.001240249,
                0.006390329,
                0.002081773,
                0.002256508,
                0.001897075,
                9.97603E-05,
                6.65068E-05,
                0.000335,
                0.000999373,
                0.000471326,
                0.000653666,
                0.000896488,
                0.002751133,
                0.001834089,
                0.003668178,
                0.013129695,
                0.015725228,
                0.017045135,
                0.014330059,
                0.001472731,
                0.000981821,
                0.001963642,
                0.004080736,
                0.006958043,
                0.009649877,
                0.013234587,
                0.00036181,
                0.000241207,
                0.001237053,
                0.006378891,
                0.002068074,
                0.002241659,
                0.001884591,
                9.84773E-05,
                6.56515E-05,
                0.000333289,
                0.000995818,
                0.000465264,
                0.000645259,
                0.000884959,
                0.012211473,
                0.008140982,
                0.016281963,
                0.058278861,
                0.069799674,
                0.075658351,
                0.063606928,
                0.006537022,
                0.004358015,
                0.008716029,
                0.018113189,
                0.030884712,
                0.042832973,
                0.058744446};
            double avgLengths[] = {5.55, 2.78, 5.24};
            double suddenDeathProbs[] = {3.79825E-07,
                3.79825E-07,
                3.79825E-07,
                3.84118E-06,
                2.00505E-05,
                3.47985E-05,
                0.000105441,
                2.03327E-07,
                2.03327E-07,
                2.03327E-07,
                1.19385E-06,
                8.87187E-06,
                1.97007E-05,
                9.73804E-05};

            uint8_t age = meta.getAge();
            uint8_t ageGroup = 0;
            if (age < 5) {
                ageGroup = 0;
            } else if (age < 15) {
                ageGroup = 1;
            } else if (age < 30) {
                ageGroup = 2;
            } else if (age < 60) {
                ageGroup = 3;
            } else if (age < 70) {
                ageGroup = 4;
            } else if (age < 80) {
                ageGroup = 5;
            } else {
                ageGroup = 6;
            }
            bool sex = meta.getSex();
            uint8_t type = meta.getPrecondIdx() == 2 ? 1 : (meta.getPrecondIdx() == 4 ? 2 : 0);

            if (timestamp > 0 && agentStat.hospitalizedUntilTimestamp == timestamp) {
                if (RandomGenerator::randomReal(1.0) < randomHospProbs[type * 4 * 7 + 2 * 7 + !sex * 7 + ageGroup]) {
                    agentStat.worstState = ppstate.die(false);
                    agentStat.worstStateTimestamp = timestamp;
                    if (agentID == tracked) {
                        printf("Agent %d died at the end of hospital stay %d\n", tracked, timestamp);
                    }
                    return;
                }
                if (agentID == tracked) {
                    printf("Agent %d recovered at the end of hospital stay %d\n", tracked, timestamp);
                }
            }

            if (ppstate.getWBState() == states::WBStates::D ||
                ppstate.getWBState() == states::WBStates::S ||
                timestamp < agentStat.hospitalizedUntilTimestamp) {
                return;
            }

            if (RandomGenerator::randomReal(1.0) < suddenDeathProbs[!sex * 7 + ageGroup] && false) {
                agentStat.worstState = ppstate.die(false);
                agentStat.worstStateTimestamp = timestamp;
                if (agentID == tracked) {
                    printf(
                        "Agent %d (%s, age %d) died of sudden death, timestamp %d\n", tracked, sex ? "M" : "F", (int)age, timestamp);
                }
                return;
            }

            int hospitalOccupancyYearlyMultDay[] = {99, 99 + 365, 99 + 2 * 365, 99 + 3 * 365};
            double hospitalOccupancyYearlyMult[] = {0.82, 0.67, 0.9, 0.9};
            int day = timestamp / (24 * 60 / timeStep);
            int d = 0;
            while (day > hospitalOccupancyYearlyMultDay[d] && d < 4) {
                d++;
            }

            double probability = randomHospProbs[type * 4 * 7 + !sex * 7 + ageGroup] *
                (probabilityMul > 0.0f ? probabilityMul : hospitalOccupancyYearlyMult[d]);
            if (RandomGenerator::randomReal(1.0) < probability) {
                unsigned avgLength = avgLengths[type];
                double p = 1.0 / (double)avgLength;
                unsigned length = RandomGenerator::geometric(p);
                if (length == 0) {
                    length = 1;
                }
                agentStat.hospitalizedTimestamp = timestamp;
                agentStat.hospitalizedUntilTimestamp = timestamp + length * 24 * 60 / timeStep;
                if (agentID == tracked) {
                    printf(
                        "Agent %d (%s, age %d) hospitalized for non-COVID disease, timestamp %d-%d\n",
                        agentID,
                        sex ? "M" : "F",
                        (int)age,
                        timestamp,
                        agentStat.hospitalizedUntilTimestamp);
                }
            }
        });
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
void Simulation<PositionType,
    TypeOfLocation,
    PPState,
    AgentMeta,
    MovementPolicy,
    InfectionPolicy,
    TestingPolicy,
    ClosurePolicy>::updateAgents(Timehandler& simTime) {
    PROFILE_FUNCTION();
    auto& ppstates = agents->PPValues;
    auto& agentStats = agents->agentStats;
    auto& agentMeta = agents->agentMetaData;
    auto& diagnosed = agents->diagnosed;
    unsigned timestamp = simTime.getTimestamp();
    unsigned tracked = locs->tracked;
    unsigned timeStepL = timeStep;
    float progressionScaling[MAX_STRAINS];
    float deathScaling[MAX_STRAINS];
    assert(diseaseProgressionScaling.size() <= MAX_STRAINS);
    for (int i = 0; i < diseaseProgressionScaling.size(); i++) {
        progressionScaling[i] = diseaseProgressionScaling[i];
    }
    for (int i = 0; i < diseaseProgressionDeathScaling.size(); i++) {
        deathScaling[i] = diseaseProgressionDeathScaling[i];
    }

    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(
            ppstates.begin(),
            agentMeta.begin(),
            agentStats.begin(),
            diagnosed.begin(),
            thrust::make_counting_iterator<unsigned>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(
            ppstates.end(),
            agentMeta.end(),
            agentStats.end(),
            diagnosed.end(),
            thrust::make_counting_iterator<unsigned>(0) + ppstates.size())),
        [timestamp, tracked, progressionScaling, deathScaling, timeStepL] HD(
            thrust::tuple<PPState&, AgentMeta&, AgentStats&, bool&, unsigned> tup) {
            auto& ppstate = thrust::get<0>(tup);
            auto& meta = thrust::get<1>(tup);
            auto& agentStat = thrust::get<2>(tup);
            auto& diagnosed = thrust::get<3>(tup);
            unsigned agentID = thrust::get<4>(tup);
            float progScaling;
            if (ppstate.getStateIdx() == 7) {
                progScaling = deathScaling[ppstate.getVariant()] / meta.getScalingSymptoms(ppstate.getVariant(), ppstate.getStateIdx());
            } else {
                progScaling = progressionScaling[ppstate.getVariant()];
            }
            bool recovered = ppstate.update(
                meta.getScalingSymptoms(ppstate.getVariant(), ppstate.getStateIdx()) * progScaling,
                agentStat,
                meta,
                timestamp,
                agentID,
                tracked,
                timeStepL);
            if (recovered) {
                diagnosed = false;
            }
        });
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
void Simulation<PositionType,
    TypeOfLocation,
    PPState,
    AgentMeta,
    MovementPolicy,
    InfectionPolicy,
    TestingPolicy,
    ClosurePolicy>::flagHealthcareWorkers() {
    auto& ppstates = agents->PPValues;
    auto& agentStats = agents->agentStats;

    healthcareWorker.resize(ppstates.size());
    auto* locationOffsetPtr = thrust::raw_pointer_cast(agents->locationOffset.data());
    auto* possibleTypesPtr = thrust::raw_pointer_cast(agents->possibleTypes.data());
    auto* locationTypePtr = thrust::raw_pointer_cast(locs->locType.data());
    auto* possibleLocationsPtr = thrust::raw_pointer_cast(agents->possibleLocations.data());
    auto healthworker = [locationOffsetPtr, possibleTypesPtr, possibleLocationsPtr, locationTypePtr] HD(
                            unsigned id) -> bool {
        for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
            if (possibleTypesPtr[idx] == 4 &&
                (locationTypePtr[possibleLocationsPtr[idx]] == 12 || locationTypePtr[possibleLocationsPtr[idx]] == 14)) {
                return true;
            }
        }
        return false;
    };

    thrust::transform(
        thrust::make_counting_iterator(unsigned(0)),
        thrust::make_counting_iterator(unsigned(agentStats.size())),
        healthcareWorker.begin(),
        healthworker);

    healthcareWorkerCount = thrust::count(healthcareWorker.begin(), healthcareWorker.end(), true);
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
void Simulation<PositionType,
    TypeOfLocation,
    PPState,
    AgentMeta,
    MovementPolicy,
    InfectionPolicy,
    TestingPolicy,
    ClosurePolicy>::setupHospitalizations(const cxxopts::ParseResult& result) {
    dueWithCOVID = result["dueWithCOVID"].as<unsigned>();
    std::string countStr = result["totalHospitalizations"].as<std::string>();
    std::ifstream t(countStr.c_str());
    std::string buffer;

    while (std::getline(t, buffer)) {
        if (buffer.length() == 0 || buffer.at(0) == '#') {
            continue;
        }
        totalHospitalizations = splitStringInt(buffer, ',');
    }
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
Simulation<PositionType,
    TypeOfLocation,
    PPState,
    AgentMeta,
    MovementPolicy,
    InfectionPolicy,
    TestingPolicy,
    ClosurePolicy>::Simulation(const cxxopts::ParseResult& result)
    : timeStep(result["deltat"].as<decltype(timeStep)>()),
      lengthOfSimulationWeeks(result["weeks"].as<decltype(lengthOfSimulationWeeks)>()),
      simTime(timeStep, 0, static_cast<Days>(result["startDay"].as<unsigned>()), result["startDate"].as<unsigned>()) {
    PROFILE_FUNCTION();
    outAgentStat = result["outAgentStat"].as<std::string>();
    enableOtherDisease = result["otherDisease"].as<int>();
    infectiousnessMultiplier = splitStringFloat(result["infectiousnessMultiplier"].as<std::string>(), ',');
    diseaseProgressionScaling = splitStringFloat(result["diseaseProgressionScaling"].as<std::string>(), ',');
    diseaseProgressionDeathScaling = splitStringFloat(result["diseaseProgressionDeathScaling"].as<std::string>(), ',');
    diagnosticLevel = result["diags"].as<unsigned>();
    setupHospitalizations(result);
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
    if (omp_get_max_threads() == 1) {
        Util::needAgentsSortedByLocation = 0;
    }
#endif
    InfectionPolicy<Simulation>::initializeArgs(result);
    MovementPolicy<Simulation>::initializeArgs(result);
    TestingPolicy<Simulation>::initializeArgs(result);
    ClosurePolicy<Simulation>::initializeArgs(result);
    immunization = new Immunization<Simulation>(this);
    immunization->initializeArgs(result);
    agents->initializeArgs(result);
    BEGIN_PROFILING("DataProvider");
    DataProvider data{ result };
    END_PROFILING("DataProvider");
    try {
        std::string header = PPState_t::initTransitionMatrix(
            data.acquireProgressionMatrices(), data.acquireProgressionConfig(), infectiousnessMultiplier);
        agents->initAgentMeta(data.acquireParameters());
        locs->initLocationTypes(data.acquireLocationTypes());
        auto tmp = locs->initLocations(data.acquireLocations(), data.acquireLocationTypes());
        auto cemeteryID = tmp.first;
        auto locationMapping = tmp.second;
        locs->initializeArgs(result);
        MovementPolicy<Simulation>::init(data.acquireLocationTypes(), cemeteryID);
        this->hospitalType = data.acquireLocationTypes().hospital;
        TestingPolicy<Simulation>::init(data.acquireLocationTypes());
        auto agentTypeMapping = agents->initAgentTypes(data.acquireAgentTypes());
        agents->initAgents(
            data.acquireAgents(),
            locationMapping,
            agentTypeMapping,
            data.getAgentTypeLocTypes(),
            data.acquireProgressionMatrices(),
            data.acquireLocationTypes());
        RandomGenerator::resize(agents->PPValues.size());
        statesHeader = header + "H\tT\tP1\tP2\tQ\tQT\tNQ\tMUT\tHOM\tVAC\tNI\tINF\tREINF\tBSTR\tIMM\tHCI\tHCE\tINFV\tINFH\tVNI";
        std::cout << statesHeader << '\n';
        auto locTypes = data.acquireLocationTypes();
        homeType = locTypes.home;
        ClosurePolicy<Simulation>::init(locTypes, data.acquireClosureRules(), statesHeader);
        locs->initialize();
        immunization->initCategories();
        flagHealthcareWorkers();
    } catch (const CustomErrors& e) {
        std::cerr << e.what() << '\n';
        succesfullyInitialized = false;
    }
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
std::vector<unsigned> Simulation<PositionType,
    TypeOfLocation,
    PPState,
    AgentMeta,
    MovementPolicy,
    InfectionPolicy,
    TestingPolicy,
    ClosurePolicy>::refreshAndPrintStatistics(Timehandler& simTime, bool print) {
    PROFILE_FUNCTION();
    auto result = locs->refreshAndGetStatistic();
    if (print) {
        for (auto val : result) {
            std::cout << val << "\t";
        }
    }

    auto& ppstates = agents->PPValues;
    auto& diagnosed = agents->diagnosed;
    auto& agentStats = agents->agentStats;
    unsigned timestamp = simTime.getTimestamp();
    unsigned hospitalized = thrust::count_if(
        thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(), agentStats.begin(), diagnosed.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(), agentStats.end(), diagnosed.end())),
        [timestamp] HD(thrust::tuple<PPState, AgentStats, bool> tup) {
            auto ppstate = thrust::get<0>(tup);
            auto agentStat = thrust::get<1>(tup);
            if (ppstate.getWBState() != states::WBStates::D &&
                ppstate.getWBState() != states::WBStates::S &&
                timestamp < agentStat.hospitalizedUntilTimestamp) {
                return true;
            }
            return false;
        });

    unsigned stayedHome = thrust::count(agents->stayedHome.begin(), agents->stayedHome.end(), true);
    std::vector<unsigned> stats(result);
    stats.push_back(hospitalized);
    if (print) {
        std::cout << hospitalized << "\t";
    }

    auto tests = TestingPolicy<Simulation>::getStats();
    if (print) {
        std::cout << thrust::get<0>(tests) << "\t" << thrust::get<1>(tests) << "\t" << thrust::get<2>(tests) << "\t";
    }
    stats.push_back(thrust::get<0>(tests));
    stats.push_back(thrust::get<1>(tests));
    stats.push_back(thrust::get<2>(tests));

    auto quarant = agents->getQuarantineStats(timestamp);
    if (print) {
        std::cout << thrust::get<0>(quarant) << "\t" << thrust::get<1>(quarant) << "\t" << thrust::get<2>(quarant) << "\t";
    }
    stats.push_back(thrust::get<0>(quarant));
    stats.push_back(thrust::get<1>(quarant));
    stats.push_back(thrust::get<2>(quarant));

    if (infectiousnessMultiplier.size() > 1) {
        unsigned allInfected = thrust::count_if(ppstates.begin(), ppstates.end(), [] HD(PPState state) { return state.isInfected(); });
        std::vector<unsigned> variantcounts(infectiousnessMultiplier.size() - 1);
        std::string out;
        for (int variant = 0; variant < infectiousnessMultiplier.size() - 1; variant++) {
            variantcounts[variant] = thrust::count_if(ppstates.begin(), ppstates.end(), [variant] HD(PPState state) {
                return state.isInfected() && state.getVariant() == variant + 1;
            });
            variantcounts[variant] = unsigned(double(variantcounts[variant]) / double(allInfected) * 100.0);
            if (variant > 0) {
                out += "," + std::to_string(variantcounts[variant]);
            } else {
                out = std::to_string(variantcounts[variant]);
            }
            stats.push_back(variantcounts[variant]);
        }
        if (print) {
            std::cout << out << "\t";
        }
    } else {
        if (print) {
            std::cout << unsigned(0) << "\t";
        }
        stats.push_back(unsigned(0));
    }

    stayedHome = stayedHome - stats[10] - stats[11];
    if (print) {
        std::cout << stayedHome << "\t";
    }
    stats.push_back(stayedHome);

    stats.push_back(immunization->immunizedToday);
    if (print) {
        std::cout << immunization->immunizedToday << "\t";
    }

    unsigned timeStepL = timeStep;
    unsigned newInfected = thrust::count_if(agentStats.begin(), agentStats.end(), [timestamp, timeStepL] HD(AgentStats agentStat) {
        return agentStat.infectedTimestamp > timestamp - 24 * 60 / timeStepL && agentStat.infectedTimestamp <= timestamp;
    });
    stats.push_back(newInfected);
    if (print) {
        std::cout << newInfected << "\t";
    }

    unsigned infectionCount =
        thrust::count_if(agentStats.begin(), agentStats.end(), [] HD(AgentStats agentStat) { return agentStat.infectedCount > 0; });
    stats.push_back(infectionCount);
    if (print) {
        std::cout << infectionCount << "\t";
    }

    unsigned reinfectionCount = thrust::transform_reduce(
        agentStats.begin(),
        agentStats.end(),
        [] HD(AgentStats agentStat) { return unsigned(agentStat.infectedCount > 1 ? agentStat.infectedCount - 1 : 0); },
        unsigned(0),
        thrust::plus<unsigned>());
    stats.push_back(reinfectionCount);
    if (print) {
        std::cout << reinfectionCount << "\t";
    }

    unsigned boosters = thrust::transform_reduce(
        agentStats.begin(),
        agentStats.end(),
        [] HD(AgentStats agentStat) { return unsigned(agentStat.immunizationCount > 1 ? agentStat.immunizationCount - 1 : 0); },
        unsigned(0),
        thrust::plus<unsigned>());
    stats.push_back(boosters);
    if (print) {
        std::cout << boosters << "\t";
    }

    std::string out;
    for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++) {
        float susceptib = thrust::transform_reduce(
            ppstates.begin(),
            ppstates.end(),
            [variant] HD(PPState state) { return 1.0f - state.getSusceptible(variant); },
            0.0f,
            thrust::plus<float>());
        if (variant == 0) {
            out = std::to_string(unsigned(susceptib));
        } else {
            out += "," + std::to_string(unsigned(susceptib));
        }
        stats.push_back(unsigned(susceptib));
    }
    if (print) {
        std::cout << out << "\t";
    }

    auto& quarantined = agents->quarantined;
    unsigned infectedHCWorker = thrust::count_if(
        thrust::make_zip_iterator(thrust::make_tuple(healthcareWorker.begin(), ppstates.begin(), quarantined.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(healthcareWorker.end(), ppstates.end(), quarantined.end())),
        [] HD(thrust::tuple<bool, PPState, bool> tup) {
            return thrust::get<0>(tup) && (thrust::get<1>(tup).isInfected() || thrust::get<2>(tup));
        });
    unsigned exposedHCWorker = thrust::count_if(
        thrust::make_zip_iterator(thrust::make_tuple(healthcareWorker.begin(), agentStats.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(healthcareWorker.end(), agentStats.end())),
        [timestamp, timeStepL] HD(thrust::tuple<bool, AgentStats> tup) {
            return thrust::get<0>(tup) &&
                (thrust::get<1>(tup).infectedTimestamp > timestamp - 24 * 60 / timeStepL &&
                    thrust::get<1>(tup).infectedTimestamp <= timestamp);
        });
    stats.push_back((unsigned)(infectedHCWorker * 100) / MAX(1, healthcareWorkerCount));
    stats.push_back((unsigned)exposedHCWorker);
    if (print) {
        std::cout << (unsigned)(infectedHCWorker * 100) / MAX(1, healthcareWorkerCount) << "\t";
        std::cout << (unsigned)exposedHCWorker << "\t";
    }

    unsigned countInf;
    for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++) {
        countInf = thrust::count_if(agentStats.begin(), agentStats.end(), [variant] HD(AgentStats stats) {
            return ((1 << variant) & stats.variant) ? true : false;
        });
        if (variant == 0) {
            out = std::to_string(unsigned(countInf));
        } else {
            out += "," + std::to_string(unsigned(countInf));
        }
        stats.push_back((unsigned)countInf);
    }
    if (print) {
        std::cout << out << "\t";
    }

    unsigned hospitalTypeLocal = hospitalType;
    unsigned infInHosp = thrust::count_if(
        thrust::make_zip_iterator(
            thrust::make_tuple(ppstates.begin(), thrust::make_permutation_iterator(locs->locType.begin(), agents->location.begin()))),
        thrust::make_zip_iterator(
            thrust::make_tuple(ppstates.end(), thrust::make_permutation_iterator(locs->locType.begin(), agents->location.end()))),
        [hospitalTypeLocal] HD(thrust::tuple<PPState, unsigned> tup) {
            return (thrust::get<0>(tup).isInfected() && thrust::get<1>(tup) == hospitalTypeLocal);
        });
    stats.push_back(infInHosp);
    if (print) {
        std::cout << infInHosp << "\t";
    }

    unsigned vaccNotInf = thrust::count_if(agentStats.begin(), agentStats.end(), [] HD(AgentStats agentStat) {
        return agentStat.infectedCount == 0 && agentStat.immunizationCount > 0;
    });
    stats.push_back(vaccNotInf);
    if (print) {
        std::cout << vaccNotInf << '\n';
    }

    return stats;
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
std::vector<unsigned> Simulation<PositionType,
    TypeOfLocation,
    PPState,
    AgentMeta,
    MovementPolicy,
    InfectionPolicy,
    TestingPolicy,
    ClosurePolicy>::countVariantCases() {
    auto& ppstates = agents->PPValues;
    std::vector<unsigned> variantCounts;
    for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++) {
        variantCounts.push_back(thrust::count_if(ppstates.begin(), ppstates.end(), [variant] HD(PPState state) {
            return state.isInfected() && state.getVariant() == variant;
        }));
    }
    return variantCounts;
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
void Simulation<PositionType, TypeOfLocation, PPState, AgentMeta, MovementPolicy, InfectionPolicy, TestingPolicy, ClosurePolicy>::processFlags(
    char **argv, int argc) {
    for (int i = 0; i < argc; i++) {
        std::string flag = argv[i];
        std::string prefix = flag.substr(0, 2);

        if (prefix == "TP") {
            if (diagnosticLevel > 0) {
                std::cout << "Testing policy " << flag << std::endl;
            }
            if (flag == "TPdef") {
                updateTestingProbs("0.00005,0.01,0.0005,0.0005,0.005,0.05");
            } else if (flag == "TP015") {
                updateTestingProbs("0.00005,0.2,0.04,0.04,0.005,0.05");
            } else if (flag == "TP035") {
                updateTestingProbs("0.00005,1.0,0.2,0.2,0.005,0.05");
            }
        } else if (prefix == "PL") {
            if (diagnosticLevel > 0) {
                std::cout << "Location closures " << flag << std::endl;
            }
            if (flag == "PLNONE" || flag == "PL0") {
                const bool shouldOpen = flag == "PLNONE";
                int fixListArr[4] = {5, 6, 22, 44};
                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(locs->locType.begin(), locs->states.begin(), locs->essential.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(locs->locType.end(), locs->states.end(), locs->essential.end())),
                    [fixListArr, shouldOpen] HD(thrust::tuple<unsigned&, bool&, uint8_t&> tup) {
                        auto& type = thrust::get<0>(tup);
                        auto& isOpen = thrust::get<1>(tup);
                        auto& isEssential = thrust::get<2>(tup);
                        if (isEssential == 1) {
                            return;
                        }
                        for (unsigned idx = 0; idx < 4; idx++) {
                            if (type == fixListArr[idx]) {
                                isOpen = shouldOpen;
                            }
                        }
                    });
            }
        } else if (prefix == "CF") {
            if (diagnosticLevel > 0) {
                std::cout << "Curfew policy: " << flag << std::endl;
            }
            if (flag == "CFNONE") {
                toggleCurfew(false, 0, 0);
            } else if (flag == "CF2000-0500") {
                toggleCurfew(true, 20 * 60, 5 * 60);
            }
        } else if (prefix == "SO") {
            if (diagnosticLevel > 0) {
                std::cout << "School age restriction: " << flag << std::endl;
            }
            if (flag == "SONONE") {
                setSchoolAgeRestriction(99);
            } else if (flag == "SO12") {
                setSchoolAgeRestriction(12);
            } else if (flag == "SO3") {
                setSchoolAgeRestriction(3);
            }
        } else if (prefix == "QU") {
            if (diagnosticLevel > 0) {
                std::cout << "Quarantine policy: " << flag << std::endl;
            }
            if (flag == "QU0") {
                quarantinePolicy(0);
            } else if (flag == "QU1") {
                quarantinePolicy(1);
            } else if (flag == "QU2") {
                quarantinePolicy(2);
            } else if (flag == "QU3") {
                quarantinePolicy(3);
            }
        } else if (prefix == "MA") {
            if (flag == "MA1.0") {
                if (currentMaskValue != 1.0) {
                    if (diagnosticLevel > 0) {
                        std::cout << "Masks at 100%" << std::endl;
                    }
                    double currentMaskValue_local = currentMaskValue;
                    int homeType_l = homeType;
                    thrust::for_each(
                        thrust::make_zip_iterator(thrust::make_tuple(locs->locType.begin(), locs->infectiousness.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(locs->locType.end(), locs->infectiousness.end())),
                        [currentMaskValue_local, homeType_l] HD(thrust::tuple<unsigned&, float&> tup) {
                            auto& type = thrust::get<0>(tup);
                            auto& infectiousness = thrust::get<1>(tup);
                            if (type != homeType_l) {
                                infectiousness = infectiousness / currentMaskValue_local;
                            }
                        });
                    currentMaskValue = 1.0;
                }
            } else if (flag == "MA0.8") {
                if (currentMaskValue != 0.8) {
                    if (diagnosticLevel > 0) {
                        std::cout << "Masks at 80%" << std::endl;
                    }
                    currentMaskValue = 0.8;
                    double currentMaskValue_local = currentMaskValue;
                    int homeType_l = homeType;
                    thrust::for_each(
                        thrust::make_zip_iterator(thrust::make_tuple(locs->locType.begin(), locs->infectiousness.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(locs->locType.end(), locs->infectiousness.end())),
                        [currentMaskValue_local, homeType_l] HD(thrust::tuple<unsigned&, float&> tup) {
                            auto& type = thrust::get<0>(tup);
                            auto& infectiousness = thrust::get<1>(tup);
                            if (type != homeType_l) {
                                infectiousness = infectiousness * currentMaskValue_local;
                            }
                        });
                }
            }
        }
    }
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
void Simulation<PositionType, TypeOfLocation, PPState, AgentMeta, MovementPolicy, InfectionPolicy, TestingPolicy, ClosurePolicy>::runSimulation() {
    std::vector<unsigned> variantCounts;
    if (!succesfullyInitialized) {
        return;
    }
    PROFILE_FUNCTION();
    const Timehandler endOfSimulation(timeStep, lengthOfSimulationWeeks, Days::MONDAY);
    while (simTime < endOfSimulation) {
        if (simTime.isMidnight()) {
            BEGIN_PROFILING("midnight")
            if (simTime.getTimestamp() > 0) {
                TestingPolicy<Simulation>::performTests(simTime, timeStep);
            }
            if (simTime.getTimestamp() > 0) {
                updateAgents(simTime);
            }
            immunization->update(simTime, timeStep);
            if (enableOtherDisease) {
                otherDisease(simTime, timeStep);
            }
            auto stats = refreshAndPrintStatistics(simTime);
            ClosurePolicy<Simulation>::midnight(simTime, timeStep, stats);
            MovementPolicy<Simulation>::planLocations(simTime, timeStep);
            variantCounts = countVariantCases();
            END_PROFILING("midnight")
        }
        MovementPolicy<Simulation>::movement(simTime, timeStep);
        ClosurePolicy<Simulation>::step(simTime, timeStep);
        for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++) {
            if (variantCounts[variant] > 0) {
                InfectionPolicy<Simulation>::infectionsAtLocations(simTime, timeStep, variant);
            }
        }
        ++simTime;
    }
    agents->printAgentStatJSON(outAgentStat);
    InfectionPolicy<Simulation>::finalize();
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
std::vector<unsigned> Simulation<PositionType,
    TypeOfLocation,
    PPState,
    AgentMeta,
    MovementPolicy,
    InfectionPolicy,
    TestingPolicy,
    ClosurePolicy>::runForDay(int argc, char **args) {
    std::vector<unsigned> variantCounts;

    PROFILE_FUNCTION();
    processFlags(args, argc);
    auto stats = refreshAndPrintStatistics(simTime, false);
    ClosurePolicy<Simulation>::midnight(simTime, timeStep, stats);
    MovementPolicy<Simulation>::planLocations(simTime, timeStep);
    variantCounts = countVariantCases();

    unsigned stepsPerDay = simTime.getStepsPerDay();
    for (unsigned i = 0; i < stepsPerDay; i++) {
        MovementPolicy<Simulation>::movement(simTime, timeStep);
        ClosurePolicy<Simulation>::step(simTime, timeStep);
        for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++) {
            if (variantCounts[variant] > 0) {
                InfectionPolicy<Simulation>::infectionsAtLocations(simTime, timeStep, variant);
            }
        }
        ++simTime;
    }

    BEGIN_PROFILING("midnight")
    if (simTime.getTimestamp() > 0) {
        TestingPolicy<Simulation>::performTests(simTime, timeStep);
    }
    if (simTime.getTimestamp() > 0) {
        updateAgents(simTime);
    }
    immunization->update(simTime, timeStep);
    if (enableOtherDisease) {
        otherDisease(simTime, timeStep);
    }
    stats = refreshAndPrintStatistics(simTime);
    END_PROFILING("midnight")

    return stats;
}

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy,
    template<typename>
    typename ClosurePolicy>
void Simulation<PositionType, TypeOfLocation, PPState, AgentMeta, MovementPolicy, InfectionPolicy, TestingPolicy, ClosurePolicy>::finalize() {
    agents->printAgentStatJSON(outAgentStat);
    InfectionPolicy<Simulation>::finalize();
}

template class Simulation<config::PositionType,
    config::TypeOfLocation,
    config::PPStates,
    BasicAgentMeta,
    RealMovement,
    BasicInfection,
    DetailedTesting,
    RuleClosure>;
