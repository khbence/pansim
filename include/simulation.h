#pragma once
#include "datatypes.h"
#include "agentsList.h"
#include "locationList.h"
#include "timeHandler.h"
#include "customExceptions.h"
#include "globalStates.h"
#include "statistics.h"
#include "timing.h"
#include "util.h"
#include <cxxopts.hpp>
#include "dataProvider.h"
#include "immunization.h"

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
class Simulation
    : private MovementPolicy<Simulation<PositionType,
          TypeOfLocation,
          PPState,
          AgentMeta,
          MovementPolicy,
          InfectionPolicy,
          TestingPolicy,
          ClosurePolicy>>
    , InfectionPolicy<Simulation<PositionType,
          TypeOfLocation,
          PPState,
          AgentMeta,
          MovementPolicy,
          InfectionPolicy,
          TestingPolicy,
          ClosurePolicy>>
    , TestingPolicy<Simulation<PositionType,
          TypeOfLocation,
          PPState,
          AgentMeta,
          MovementPolicy,
          InfectionPolicy,
          TestingPolicy,
          ClosurePolicy>>
    , ClosurePolicy<Simulation<PositionType,
          TypeOfLocation,
          PPState,
          AgentMeta,
          MovementPolicy,
          InfectionPolicy,
          TestingPolicy,
          ClosurePolicy>> {

public:
    using PPState_t = PPState;
    using AgentMeta_t = AgentMeta;
    using LocationType = LocationsList<Simulation>;
    using PositionType_t = PositionType;
    using TypeOfLocation_t = TypeOfLocation;
    using AgentListType = AgentList<PPState_t, AgentMeta_t, LocationType>;

    // private:
    AgentListType* agents = AgentListType::getInstance();
    LocationType* locs = LocationType::getInstance();
    unsigned timeStep;
    unsigned lengthOfSimulationWeeks;
    bool succesfullyInitialized = true;
    std::string outAgentStat;
    std::string statesHeader;
    int enableOtherDisease = 1;
    Immunization<Simulation> *immunization;
    float mutationMultiplier;
    float mutationProgressionScaling=1.0;
    Timehandler simTime;

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;
    friend class TestingPolicy<Simulation>;
    friend class ClosurePolicy<Simulation>;

    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("otherDisease",
            "Enable (1) or disable (0) non-COVID related hospitalization and sudden death ",
            cxxopts::value<int>()->default_value("1"))
            ("mutationMultiplier",
            "infectiousness multiplier for mutated virus ",
            cxxopts::value<float>()->default_value("1.0"))
            ("mutationProgressionScaling",
            "disease progression scaling for mutated virus ",
            cxxopts::value<float>()->default_value("1.0"))
            ("startDay",
            "day of the week to start the simulation with (Monday is 0) ",
            cxxopts::value<unsigned>()->default_value("0"));
            
        InfectionPolicy<Simulation>::addProgramParameters(options);
        MovementPolicy<Simulation>::addProgramParameters(options);
        TestingPolicy<Simulation>::addProgramParameters(options);
        ClosurePolicy<Simulation>::addProgramParameters(options);
        AgentListType::addProgramParameters(options);
        Immunization<Simulation>::addProgramParameters(options);
    }

    void otherDisease(Timehandler& simTime, unsigned timeStep) {
        //PROFILE_FUNCTION();
        auto& ppstates = agents->PPValues;
        auto& agentStats = agents->agentStats;
        auto& agentMeta = agents->agentMetaData;
        unsigned timestamp = simTime.getTimestamp();
        unsigned tracked = locs->tracked;
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                             agentMeta.begin(),
                             agentStats.begin(),
                             thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                agentMeta.end(),
                agentStats.end(),
                thrust::make_counting_iterator<unsigned>(0) + ppstates.size())),
            [timestamp, tracked, timeStep] HD(
                thrust::tuple<PPState&, AgentMeta&, AgentStats&, unsigned> tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                auto& agentStat = thrust::get<2>(tup);
                unsigned agentID = thrust::get<3>(tup);

                if (ppstate.getWBState() == states::WBStates::D) return;

                //
                // Parameters - to be extracted to config
                //
                //0-4ig	     5-14ig	      15-29ig	   30-59ig	    60-69ig	     70-79ig	  80+
                double randomHospProbs[] = {
                0.00017785, 0.000118567, 0.000614453, 0.003174864, 0.001016573, 0.0011019, 0.000926381,     //m general hospitalization
                9.52062E-05, 6.34708E-05, 0.000328928, 0.000986754, 0.00044981, 0.000623826, 0.000855563,   //f      
                0.006333678, 0.004222452, 0.008444904, 0.030227275, 0.036202732, 0.03924143, 0.032990764,   //m        death
                0.003390532, 0.002260355, 0.00452071, 0.009394699, 0.016018856, 0.022216016, 0.030468759,   //f
                0.000364207, 0.000242805, 0.001240249, 0.006390329, 0.002081773, 0.002256508, 0.001897075,  //m cardiov hospitalization
                9.97603E-05, 6.65068E-05, 0.000335, 0.000999373, 0.000471326, 0.000653666, 0.000896488,     //f
                0.002751133, 0.001834089, 0.003668178, 0.013129695, 0.015725228, 0.017045135, 0.014330059,  //m        death
                0.001472731, 0.000981821, 0.001963642, 0.004080736, 0.006958043, 0.009649877, 0.013234587,  //f
                0.00036181, 0.000241207, 0.001237053, 0.006378891, 0.002068074, 0.002241659, 0.001884591,   //m pulmon  hospitalization
                9.84773E-05, 6.56515E-05, 0.000333289, 0.000995818, 0.000465264, 0.000645259, 0.000884959,  //f
                0.012211473, 0.008140982, 0.016281963, 0.058278861, 0.069799674, 0.075658351, 0.063606928,  //m        death
                0.006537022, 0.004358015, 0.008716029, 0.018113189, 0.030884712, 0.042832973, 0.058744446 };//f
                double avgLengths[] = {5.55, 2.78, 5.24};
                //0-4ig	     5-14ig	      15-29ig	   30-59ig	    60-69ig	     70-79ig	  80+
                double suddenDeathProbs[] = {
                    3.79825E-07, 3.79825E-07, 3.79825E-07, 3.84118E-06, 2.00505E-05, 3.47985E-05, 0.000105441,
                    2.03327E-07, 2.03327E-07, 2.03327E-07, 1.19385E-06, 8.87187E-06, 1.97007E-05, 9.73804E-05 }; //female
                

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
                //precond - 2 is cardiovascular, 4 is pulmonary. All others are general
                uint8_t type = meta.getPrecondIdx() == 2 ? 1 : (meta.getPrecondIdx() == 4 ? 2 : 0);

                //
                // non-COVID hospitalization - ended, see if dies or lives
                //
                if (timestamp>0 && agentStat.hospitalizedUntilTimestamp == timestamp) {
                    if (RandomGenerator::randomReal(1.0) < randomHospProbs[type * 4 * 7 + 2 * 7 + !sex * 7 + ageGroup]) {
                        agentStat.worstState = ppstate.die(false); //not COVID-related
                        agentStat.worstStateTimestamp = timestamp;
                        //printf("Agent %d died at the end of hospital stay %d\n", agentID, timestamp);
                        if (agentID == tracked) {
                            printf("Agent %d died at the end of hospital stay %d\n", tracked, timestamp);
                        }
                        return;
                    } else {
                        //printf("Agent %d recovered at the end of hospital stay %d\n", agentID, timestamp);
                        if (agentID == tracked) {
                            printf("Agent %d recovered at the end of hospital stay %d\n", tracked, timestamp);
                        }
                    }
                }

                //
                // Sudden death
                //

                //If already dead, or in hospital (due to COVID or non-COVID), return
                if (ppstate.getWBState() == states::WBStates::S ||
                    timestamp < agentStat.hospitalizedUntilTimestamp) return;

                
                if (RandomGenerator::randomReal(1.0) < suddenDeathProbs[!sex*7+ageGroup] && false) {
                    agentStat.worstState = ppstate.die(false); //not COVID-related
                    agentStat.worstStateTimestamp = timestamp;
                    if (agentID == tracked) {
                        printf("Agent %d (%s, age %d) died of sudden death, timestamp %d\n", tracked, sex?"M":"F", (int)age, timestamp);
                    }
                    return;
                }

                //
                // Random hospitalization
                //
                double probability = randomHospProbs[type * 4 * 7 + !sex * 7 + ageGroup];
                if (RandomGenerator::randomReal(1.0) < probability) {
                    //Got hospitalized
                    //Length;
                    unsigned avgLength = avgLengths[type]; //In days
                    double p = 1.0 / (double) avgLength;
                    unsigned length = RandomGenerator::geometric(p);
                    if (length == 0) length = 1; // At least one day
                    agentStat.hospitalizedTimestamp = timestamp;
                    agentStat.hospitalizedUntilTimestamp = timestamp + length*24*60/timeStep;
                    //printf("Agent %d hospitalized for non-COVID disease, timestamp %d-%d\n", agentID, timestamp, agentStat.hospitalizedUntilTimestamp);
                    if (agentID == tracked) {
                        printf("Agent %d (%s, age %d) hospitalized for non-COVID disease, timestamp %d-%d\n", agentID, sex?"M":"F", (int)age, timestamp, agentStat.hospitalizedUntilTimestamp);
                    }
                }
            });
    }

    void updateAgents(Timehandler& simTime) {
        //PROFILE_FUNCTION();
        auto& ppstates = agents->PPValues;
        auto& agentStats = agents->agentStats;
        auto& agentMeta = agents->agentMetaData;
        auto& diagnosed = agents->diagnosed;
        unsigned timestamp = simTime.getTimestamp();
        unsigned tracked = locs->tracked;
        float progressionScaling = mutationProgressionScaling;
        // Update states
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                             agentMeta.begin(),
                             agentStats.begin(),
                             diagnosed.begin(),
                             thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                agentMeta.end(),
                agentStats.end(),
                diagnosed.end(),
                thrust::make_counting_iterator<unsigned>(0) + ppstates.size())),
            [timestamp, tracked, progressionScaling] HD(
                thrust::tuple<PPState&, AgentMeta&, AgentStats&, bool&, unsigned> tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                auto& agentStat = thrust::get<2>(tup);
                auto& diagnosed = thrust::get<3>(tup);
                unsigned agentID = thrust::get<4>(tup);
                bool recovered = ppstate.update(
                    meta.getScalingSymptoms()*(ppstate.getVariant()==1?progressionScaling:1.0), agentStat, timestamp, agentID, tracked);
                if (recovered) diagnosed = false;
            });
    }

    std::vector<unsigned> refreshAndPrintStatistics(Timehandler& simTime) {
        //PROFILE_FUNCTION();
        //COVID
        auto result = locs->refreshAndGetStatistic();
        //for (auto val : result) { std::cout << val << "\t"; }
        //non-COVID hospitalization
        auto& ppstates = agents->PPValues;
        auto& diagnosed = agents->diagnosed;
        auto& agentStats = agents->agentStats;
        unsigned timestamp = simTime.getTimestamp();
        unsigned hospitalized = 
            thrust::count_if(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(), agentStats.begin(), diagnosed.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(), agentStats.end(), diagnosed.end())),
                         [timestamp] HD (thrust::tuple<PPState, AgentStats, bool> tup) {
                             auto ppstate = thrust::get<0>(tup);
                             auto agentStat = thrust::get<1>(tup);
                             auto diagnosed = thrust::get<2>(tup);
                             if (ppstate.getWBState() != states::WBStates::D &&  //avoid double-counting with COVID
                                 ppstate.getWBState() != states::WBStates::S &&
                                 diagnosed == false &&
                                 timestamp < agentStat.hospitalizedUntilTimestamp) return true;
                             else return false;
                         });
        unsigned stayedHome = thrust::count(agents->stayedHome.begin(), agents->stayedHome.end(), true);
        std::vector<unsigned> stats(result);
        stats.push_back(hospitalized);
        //std::cout << hospitalized << "\t";
        //Testing
        auto tests = TestingPolicy<Simulation>::getStats();
        //std::cout << thrust::get<0>(tests) << "\t" << thrust::get<1>(tests) << "\t"
        //          << thrust::get<2>(tests) << "\t";
        stats.push_back(thrust::get<0>(tests));
        stats.push_back(thrust::get<1>(tests));
        stats.push_back(thrust::get<2>(tests));
        //Quarantine stats
        auto quarant = agents->getQuarantineStats(timestamp);
        //std::cout << thrust::get<0>(quarant) << "\t" << thrust::get<1>(quarant) << "\t"
        //          << thrust::get<2>(quarant) << "\t";
        stats.push_back(thrust::get<0>(quarant));
        stats.push_back(thrust::get<1>(quarant));
        stats.push_back(thrust::get<2>(quarant));

        if (mutationMultiplier != 1.0f) {
            unsigned allInfected = thrust::count_if(ppstates.begin(),
                                                 ppstates.end(),
                         [] HD (PPState state) {
                             return state.isInfected();
                         });
            unsigned variant1 = thrust::count_if(ppstates.begin(),
                                                 ppstates.end(),
                         [] HD (PPState state) {
                             return state.isInfected() && state.getVariant()==1;
                         });
            unsigned percentage = unsigned(double(variant1)/double(allInfected)*100.0);
        //    std::cout << percentage << "\t";
            stats.push_back(percentage);
        } else {
        //    std::cout << unsigned(0) << "\t";
            stats.push_back(unsigned(0));
        }

        //Stayed home count
        stayedHome = stayedHome - stats[10] - stats[11]; //Subtract dead
        //std::cout << stayedHome << "\t";
        stats.push_back(stayedHome);

        //Number of immunized
        stats.push_back(immunization->immunizedToday);
        //std::cout << immunization->immunizedToday << "\t";

        //Number of new infections
        unsigned timeStepL = timeStep;
        unsigned newInfected = 
            thrust::count_if(agentStats.begin(), agentStats.end(),
                         [timestamp,timeStepL] HD (AgentStats agentStat) {
                             return (agentStat.infectedTimestamp > timestamp - 24*60/timeStepL && agentStat.infectedTimestamp <= timestamp);
                         });
        stats.push_back(newInfected);
        //std::cout << newInfected;

        //std::cout << '\n';
        return stats;
    }

public:
    explicit Simulation(const cxxopts::ParseResult& result)
        : timeStep(result["deltat"].as<decltype(timeStep)>()),
          lengthOfSimulationWeeks(result["weeks"].as<decltype(lengthOfSimulationWeeks)>()), 
          simTime(timeStep,0,static_cast<Days>(result["startDay"].as<unsigned>())) {
        //PROFILE_FUNCTION();
        outAgentStat = result["outAgentStat"].as<std::string>();
        InfectionPolicy<Simulation>::initializeArgs(result);
        MovementPolicy<Simulation>::initializeArgs(result);
        TestingPolicy<Simulation>::initializeArgs(result);
        ClosurePolicy<Simulation>::initializeArgs(result);
        immunization = new Immunization<Simulation>(this);
        immunization->initializeArgs(result);
        agents->initializeArgs(result);
        enableOtherDisease = result["otherDisease"].as<int>();
        mutationMultiplier = result["mutationMultiplier"].as<float>();
        mutationProgressionScaling = result["mutationProgressionScaling"].as<float>();
        DataProvider data{ result };
        try {
            std::string header = PPState_t::initTransitionMatrix(
                data.acquireProgressionMatrices(), data.acquireProgressionConfig(),mutationMultiplier);
            agents->initAgentMeta(data.acquireParameters());
            locs->initLocationTypes(data.acquireLocationTypes());
            auto tmp = locs->initLocations(data.acquireLocations(), data.acquireLocationTypes());
            auto cemeteryID = tmp.first;
            auto locationMapping = tmp.second;
            locs->initializeArgs(result);
            MovementPolicy<Simulation>::init(data.acquireLocationTypes(), cemeteryID);
            TestingPolicy<Simulation>::init(data.acquireLocationTypes());
            auto agentTypeMapping = agents->initAgentTypes(data.acquireAgentTypes());
            agents->initAgents(data.acquireAgents(),
                locationMapping,
                agentTypeMapping,
                data.getAgentTypeLocTypes(),
                data.acquireProgressionMatrices(),
                data.acquireLocationTypes());
            RandomGenerator::resize(agents->PPValues.size());
            statesHeader = header + "H\tT\tP1\tP2\tQ\tQT\tNQ\tMUT\tHOM\tVAC\tNI";
            //std::cout << statesHeader << '\n';
            ClosurePolicy<Simulation>::init(data.acquireLocationTypes(), data.acquireClosureRules(), statesHeader);
        } catch (const CustomErrors& e) {
            std::cerr << e.what();
            succesfullyInitialized = false;
        }
        locs->initialize();
        immunization->initCategories();
    }

    void runSimulation() {
        if (!succesfullyInitialized) { return; }
        //PROFILE_FUNCTION();
        const Timehandler endOfSimulation(timeStep, lengthOfSimulationWeeks, Days::MONDAY);
        while (simTime < endOfSimulation) {
            //std::cout << simTime.getTimestamp() << std::endl;
            if (simTime.isMidnight()) {
                BEGIN_PROFILING("midnight");
                if (simTime.getTimestamp() > 0) TestingPolicy<Simulation>::performTests(simTime, timeStep);
                if (simTime.getTimestamp() > 0) updateAgents(simTime);// No disease progression at launch
                if (enableOtherDisease) otherDisease(simTime,timeStep);
                auto stats = refreshAndPrintStatistics(simTime);
                ClosurePolicy<Simulation>::midnight(simTime, timeStep, stats);
                MovementPolicy<Simulation>::planLocations(simTime, timeStep);
                immunization->update(simTime, timeStep);
                END_PROFILING("midnight");
            }
            BEGIN_PROFILING("movementUpper");
            MovementPolicy<Simulation>::movement(simTime, timeStep);
            END_PROFILING("movementUpper");
            BEGIN_PROFILING("step");
            ClosurePolicy<Simulation>::step(simTime, timeStep);
            END_PROFILING("step");
            BEGIN_PROFILING("infect");
            InfectionPolicy<Simulation>::infectionsAtLocations(simTime, timeStep, 0);
            if (mutationMultiplier != 1.0f)
                InfectionPolicy<Simulation>::infectionsAtLocations(simTime, timeStep, 1);
            END_PROFILING("infect");
            ++simTime;
        }
        agents->printAgentStatJSON(outAgentStat);
    }

    void toggleCurfew(bool enable, unsigned curfewBegin, unsigned curfewEnd) {
        MovementPolicy<Simulation>::enableCurfew = enable;
        MovementPolicy<Simulation>::curfewBegin = curfewBegin;
        MovementPolicy<Simulation>::curfewEnd = curfewEnd;
        MovementPolicy<Simulation>::curfewTimeConverted = false;
    }
    void setSchoolAgeRestriction(unsigned limit) {
        MovementPolicy<Simulation>::schoolAgeRestriction = limit;
    }
    void toggleHolidayMode(bool enable) {
        MovementPolicy<Simulation>::holidayModeActive = enable;
    }
    Timehandler& getSimTime() {
        return simTime;
    }
};
