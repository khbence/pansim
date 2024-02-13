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
#include "smallTools.h"

cxxopts::Options defineProgramParameters() {
    cxxopts::Options options("covid", "An agent-based epidemic simulator");
    options.add_options()("w,weeks", "Length of simulation in weeks", cxxopts::value<unsigned>()->default_value("12"))(
        "t,deltat", "Length of timestep in minutes", cxxopts::value<unsigned>()->default_value("10"))(
        "n,numagents", "Number of agents", cxxopts::value<int>()->default_value("-1"))(
        "N,numlocs", "Number of dummy locations", cxxopts::value<int>()->default_value("-1"))("P,progression",
        "Path to the config file for the progression matrices.",
        cxxopts::value<std::string>()->default_value(
            "inputConfigFiles" + separator() + "progressions" + separator() + "transition_config.json"))("a,agents",
        "Agents file, for all human being in the experiment.",
        cxxopts::value<std::string>()->default_value("inputRealExample" + separator() + "agents.json"))("A,agentTypes",
        "List and schedule of all type fo agents.",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "agentTypes.json"))("l,locations",
        "List of all locations in the simulation.",
        cxxopts::value<std::string>()->default_value("inputRealExample" + separator() + "locations.json"))("L,locationTypes",
        "List of all type of locations",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "locationTypes.json"))("p,parameters",
        "List of all general parameters for the simulation except the "
        "progression data.",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "parameters.json"))("c,configRandom",
        "Config file for random initialization.",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "configRandom.json"))("closures",
        "List of closure rules.",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "closureRules.json"))("r,randomStates",
        "Change the states from the agents file with the configRandom file's "
        "stateDistribution.")("outAgentStat",
        "name of the agent stat output file, if not set there will be no print",
        cxxopts::value<std::string>()->default_value(""))(
        "diags", "level of diagnositcs to print", cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(0))));

        
    options.add_options()("h,help", "Print usage");
    options.add_options()("version", "Print version");

    return options;
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
    unsigned hospitalType;
    Immunization<Simulation>* immunization;
    std::vector<float> infectiousnessMultiplier;
    std::vector<float> diseaseProgressionScaling;
    std::vector<float> diseaseProgressionDeathScaling;
    Timehandler simTime;
    thrust::device_vector<bool> healthcareWorker;
    unsigned healthcareWorkerCount;
    std::vector<int> totalHospitalizations;
    unsigned dueWithCOVID;
    double currentMaskValue = 1.0;
    int diagnosticLevel;
    unsigned homeType;

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;
    friend class TestingPolicy<Simulation>;
    friend class ClosurePolicy<Simulation>;

    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("otherDisease",
            "Enable (1) or disable (0) non-COVID related hospitalization and sudden death ",
            cxxopts::value<int>()->default_value("1"))
            ("infectiousnessMultiplier",
            "infectiousness multiplier for original strain and variants ",
            cxxopts::value<std::string>()->default_value("1.03,1.79,2.45,2.7,3.6"))
            ("diseaseProgressionScaling",
            "disease progression scaling for original strain and variants ",
            cxxopts::value<std::string>()->default_value("0.9,1.05,0.9,0.8,0.8"))
            ("diseaseProgressionDeathScaling",
            "disease progression scaling for death for original strain and variants ",
            cxxopts::value<std::string>()->default_value("1.0,1.03,1.3,0.6,0.6"))
            ("startDay",
            "day of the week to start the simulation with (Monday is 0) ",
            cxxopts::value<unsigned>()->default_value("2"))
            ("startDate",
            "days into the year the simulation starts with (Jan 1 is 0) ",
            cxxopts::value<unsigned>()->default_value("267"))
            ("totalHospitalizations",
            "number of agents hospitalized every day for any reason",
            cxxopts::value<std::string>()->default_value("inputConfigFiles/dailyHospitalizationTargets.txt"))
            ("dueWithCOVID",
            "total hospitalization target should be reached with 0 - unrelated+due to COVID, or 1 - unrelated+with COVID",
            cxxopts::value<unsigned>()->default_value("0"));

        InfectionPolicy<Simulation>::addProgramParameters(options);
        MovementPolicy<Simulation>::addProgramParameters(options);
        TestingPolicy<Simulation>::addProgramParameters(options);
        ClosurePolicy<Simulation>::addProgramParameters(options);
        AgentListType::addProgramParameters(options);
        Immunization<Simulation>::addProgramParameters(options);
    }

    void otherDisease(Timehandler& simTime, unsigned timeStep) {
        PROFILE_FUNCTION();
        auto& ppstates = agents->PPValues;
        auto& agentStats = agents->agentStats;
        auto& agentMeta = agents->agentMetaData;
        unsigned timestamp = simTime.getTimestamp();

        //for target unrelated Hospitalizations we need to count up those in hospital due to COVID or with COVID
        unsigned inHospitalWithCovid;
        if (dueWithCOVID == 0) { //Due to COVID, count up I5_h, I6_h, R_h
            inHospitalWithCovid = thrust::count_if(ppstates.begin(), ppstates.end(), []HD(PPState state) {
                    return (state.getStateIdx() >=6 && state.getStateIdx() <=8); //In hospital due to COVID
            });
        } else {
            //Number of people in hospital with active infection
            unsigned hospitalTypeLocal = hospitalType;
            inHospitalWithCovid = thrust::count_if(thrust::make_zip_iterator(
                thrust::make_tuple(ppstates.begin(),thrust::make_permutation_iterator(locs->locType.begin(), agents->location.begin()), agentStats.begin())),
                                                thrust::make_zip_iterator(
                thrust::make_tuple(ppstates.end(),thrust::make_permutation_iterator(locs->locType.begin(), agents->location.end()), agentStats.end())),
                [hospitalTypeLocal, timestamp] HD (thrust::tuple<PPState, unsigned, AgentStats> tup) {
                    return (thrust::get<0>(tup).isInfected() && 
                            thrust::get<1>(tup)==hospitalTypeLocal &&
                            !(thrust::get<0>(tup).getWBState() < states::WBStates::S && timestamp > 0 && thrust::get<2>(tup).hospitalizedUntilTimestamp == timestamp)); //not discharged today
                    });
        }

        float probabilityMul = 0.0f;
        unsigned simDay = simTime.getTimestamp()/simTime.getStepsPerDay();
        if (simDay < totalHospitalizations.size()) {
            //Probabilities set up based on 2019 statistics of average daily hospital occupancy of 48000
            //need to weight this probability so COVID+non-COVID gives desired target
            unsigned targetHospitalized = std::max(0.0, double(totalHospitalizations[simDay]) - double(inHospitalWithCovid)*(9730000.0/179500.0));
            probabilityMul = float(targetHospitalized)/48000.0;
            // printf("%d %g\n", simDay, probabilityMul);
        }
        unsigned tracked = locs->tracked;
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(
                ppstates.begin(), agentMeta.begin(), agentStats.begin(), thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                agentMeta.end(),
                agentStats.end(),
                thrust::make_counting_iterator<unsigned>(0) + ppstates.size())),
            [timestamp, tracked, timeStep, probabilityMul] HD(thrust::tuple<PPState&, AgentMeta&, AgentStats&, unsigned> tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                auto& agentStat = thrust::get<2>(tup);
                unsigned agentID = thrust::get<3>(tup);

                if (ppstate.getWBState() == states::WBStates::D) return;

                //
                // Parameters - to be extracted to config
                //
                // 0-4ig	     5-14ig	      15-29ig	   30-59ig	    60-69ig	     70-79ig	  80+
                double randomHospProbs[] = { 0.00017785,
                    0.000118567,
                    0.000614453,
                    0.003174864,
                    0.001016573,
                    0.0011019,
                    0.000926381,// m general hospitalization
                    9.52062E-05,
                    6.34708E-05,
                    0.000328928,
                    0.000986754,
                    0.00044981,
                    0.000623826,
                    0.000855563,// f
                    0.006333678,
                    0.004222452,
                    0.008444904,
                    0.030227275,
                    0.036202732,
                    0.03924143,
                    0.032990764,// m        death
                    0.003390532,
                    0.002260355,
                    0.00452071,
                    0.009394699,
                    0.016018856,
                    0.022216016,
                    0.030468759,// f
                    0.000364207,
                    0.000242805,
                    0.001240249,
                    0.006390329,
                    0.002081773,
                    0.002256508,
                    0.001897075,// m cardiov hospitalization
                    9.97603E-05,
                    6.65068E-05,
                    0.000335,
                    0.000999373,
                    0.000471326,
                    0.000653666,
                    0.000896488,// f
                    0.002751133,
                    0.001834089,
                    0.003668178,
                    0.013129695,
                    0.015725228,
                    0.017045135,
                    0.014330059,// m        death
                    0.001472731,
                    0.000981821,
                    0.001963642,
                    0.004080736,
                    0.006958043,
                    0.009649877,
                    0.013234587,// f
                    0.00036181,
                    0.000241207,
                    0.001237053,
                    0.006378891,
                    0.002068074,
                    0.002241659,
                    0.001884591,// m pulmon  hospitalization
                    9.84773E-05,
                    6.56515E-05,
                    0.000333289,
                    0.000995818,
                    0.000465264,
                    0.000645259,
                    0.000884959,// f
                    0.012211473,
                    0.008140982,
                    0.016281963,
                    0.058278861,
                    0.069799674,
                    0.075658351,
                    0.063606928,// m        death
                    0.006537022,
                    0.004358015,
                    0.008716029,
                    0.018113189,
                    0.030884712,
                    0.042832973,
                    0.058744446 };// f
                double avgLengths[] = { 5.55, 2.78, 5.24 };
                // 0-4ig	     5-14ig	      15-29ig	   30-59ig	    60-69ig	     70-79ig	  80+
                double suddenDeathProbs[] = { 3.79825E-07,
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
                    9.73804E-05 };// female


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
                // precond - 2 is cardiovascular, 4 is pulmonary. All others are general
                uint8_t type = meta.getPrecondIdx() == 2 ? 1 : (meta.getPrecondIdx() == 4 ? 2 : 0);

                //
                // non-COVID hospitalization - ended, see if dies or lives
                //
                if (timestamp > 0 && agentStat.hospitalizedUntilTimestamp == timestamp) {
                    if (RandomGenerator::randomReal(1.0) < randomHospProbs[type * 4 * 7 + 2 * 7 + !sex * 7 + ageGroup]) {
                        agentStat.worstState = ppstate.die(false);// not COVID-related
                        agentStat.worstStateTimestamp = timestamp;
                        // printf("Agent %d died at the end of hospital stay %d\n", agentID, timestamp);
                        if (agentID == tracked) {
                            printf("Agent %d died at the end of hospital stay %d\n", tracked, timestamp);
                        }
                        return;
                    } else {
                        // printf("Agent %d recovered at the end of hospital stay %d\n", agentID, timestamp);
                        if (agentID == tracked) {
                            printf("Agent %d recovered at the end of hospital stay %d\n", tracked, timestamp);
                        }
                    }
                }

                //
                // Sudden death
                //

                // If already dead, or in hospital (due to COVID or non-COVID), return
                if (ppstate.getWBState() == states::WBStates::D ||
                    ppstate.getWBState() == states::WBStates::S ||
                    timestamp < agentStat.hospitalizedUntilTimestamp) return;


                if (RandomGenerator::randomReal(1.0) < suddenDeathProbs[!sex * 7 + ageGroup] && false) {
                    agentStat.worstState = ppstate.die(false);// not COVID-related
                    agentStat.worstStateTimestamp = timestamp;
                    if (agentID == tracked) {
                        printf("Agent %d (%s, age %d) died of sudden death, timestamp %d\n",
                            tracked,
                            sex ? "M" : "F",
                            (int)age,
                            timestamp);
                    }
                    return;
                }

                //
                // Random hospitalization
                //
                //                                        2020   2021     2022       2023
                int hospitalOccupancyYearlyMultDay[] = {99, 99+365, 99+2*365, 99+3*365};
                double hospitalOccupancyYearlyMult[] = {0.82, 0.67, 0.9, 0.9};
                int day = timestamp / (24 * 60 / timeStep);
                int d = 0;
                while (day > hospitalOccupancyYearlyMultDay[d] && d < 4) d++;

                double probability = randomHospProbs[type * 4 * 7 + !sex * 7 + ageGroup] * (probabilityMul>0.0f ? probabilityMul : hospitalOccupancyYearlyMult[d]);
                if (RandomGenerator::randomReal(1.0) < probability) {
                    // Got hospitalized
                    // Length;
                    unsigned avgLength = avgLengths[type];// In days
                    double p = 1.0 / (double)avgLength;
                    unsigned length = RandomGenerator::geometric(p);
                    if (length == 0) length = 1;// At least one day
                    agentStat.hospitalizedTimestamp = timestamp;
                    agentStat.hospitalizedUntilTimestamp = timestamp + length * 24 * 60 / timeStep;
                    // printf("Agent %d hospitalized for non-COVID disease, timestamp %d-%d\n", agentID, timestamp,
                    // agentStat.hospitalizedUntilTimestamp);
                    if (agentID == tracked) {
                        printf("Agent %d (%s, age %d) hospitalized for non-COVID disease, timestamp %d-%d\n",
                            agentID,
                            sex ? "M" : "F",
                            (int)age,
                            timestamp,
                            agentStat.hospitalizedUntilTimestamp);
                    }
                }
            });
    }

    void updateAgents(Timehandler& simTime) {
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
        assert(diseaseProgressionScaling.size()<=MAX_STRAINS);
        for (int i = 0; i < diseaseProgressionScaling.size(); i++)
            progressionScaling[i] = diseaseProgressionScaling[i];
        for (int i = 0; i < diseaseProgressionDeathScaling.size(); i++)
            deathScaling[i] = diseaseProgressionDeathScaling[i];

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
            [timestamp, tracked, progressionScaling, deathScaling, timeStepL] HD(thrust::tuple<PPState&, AgentMeta&, AgentStats&, bool&, unsigned> tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                auto& agentStat = thrust::get<2>(tup);
                auto& diagnosed = thrust::get<3>(tup);
                unsigned agentID = thrust::get<4>(tup);
                float progScaling;
                if (ppstate.getStateIdx()==7) progScaling = deathScaling[ppstate.getVariant()]/meta.getScalingSymptoms(ppstate.getVariant(),ppstate.getStateIdx());
                else progScaling = progressionScaling[ppstate.getVariant()];
                bool recovered =
                    ppstate.update(meta.getScalingSymptoms(ppstate.getVariant(),ppstate.getStateIdx()) * progScaling,
                        agentStat,
                        meta,
                        timestamp,
                        agentID,
                        tracked, timeStepL);
                if (recovered) diagnosed = false;
            });
    }

    std::vector<unsigned> refreshAndPrintStatistics(Timehandler& simTime, bool print = true) {
        PROFILE_FUNCTION();
        // COVID
        auto result = locs->refreshAndGetStatistic();
        if (print)
            for (auto val : result) { std::cout << val << "\t"; }
        // non-COVID hospitalization
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
                auto diagnosed = thrust::get<2>(tup);
                if (ppstate.getWBState() != states::WBStates::D &&// avoid double-counting with COVID
                    ppstate.getWBState() != states::WBStates::S //&& diagnosed == false with COVID
                    && timestamp < agentStat.hospitalizedUntilTimestamp)
                    return true;
                else
                    return false;
            });
        unsigned stayedHome = thrust::count(agents->stayedHome.begin(), agents->stayedHome.end(), true);
        std::vector<unsigned> stats(result);
        stats.push_back(hospitalized);
        if (print)
            std::cout << hospitalized << "\t";
        // Testing
        auto tests = TestingPolicy<Simulation>::getStats();
        if (print)
            std::cout << thrust::get<0>(tests) << "\t" << thrust::get<1>(tests) << "\t" << thrust::get<2>(tests) << "\t";
        stats.push_back(thrust::get<0>(tests));
        stats.push_back(thrust::get<1>(tests));
        stats.push_back(thrust::get<2>(tests));
        // Quarantine stats
        auto quarant = agents->getQuarantineStats(timestamp);
        if (print)
            std::cout << thrust::get<0>(quarant) << "\t" << thrust::get<1>(quarant) << "\t" << thrust::get<2>(quarant) << "\t";
        stats.push_back(thrust::get<0>(quarant));
        stats.push_back(thrust::get<1>(quarant));
        stats.push_back(thrust::get<2>(quarant));

        if (infectiousnessMultiplier.size() > 1) {
            unsigned allInfected =
                thrust::count_if(ppstates.begin(), ppstates.end(), [] HD(PPState state) { return state.isInfected(); });
            std::vector<unsigned> variantcounts(infectiousnessMultiplier.size()-1);
            std::string out;
            for (int variant = 0; variant < infectiousnessMultiplier.size()-1; variant++) {
                variantcounts[variant] = thrust::count_if(ppstates.begin(), ppstates.end(), [variant] HD(PPState state) {
                    return state.isInfected() && state.getVariant() == variant+1;
                });
                variantcounts[variant] = unsigned(double(variantcounts[variant]) / double(allInfected) * 100.0);
                if (variant>0)
                    out = out + "," + std::to_string(variantcounts[variant]);
                else
                    out = std::to_string(variantcounts[variant]);
                stats.push_back(variantcounts[variant]);
            }
            if (print)
                std::cout << out << "\t";
        } else {
            if (print)
                std::cout << unsigned(0) << "\t";
            stats.push_back(unsigned(0));
        }

        // Stayed home count
        stayedHome = stayedHome - stats[10] - stats[11];// Subtract dead
        if (print)
            std::cout << stayedHome << "\t";
        stats.push_back(stayedHome);

        // Number of immunized
        stats.push_back(immunization->immunizedToday);
        if (print)
            std::cout << immunization->immunizedToday << "\t";

        // Number of new infections
        unsigned timeStepL = timeStep;
        unsigned newInfected =
            thrust::count_if(agentStats.begin(), agentStats.end(), [timestamp, timeStepL] HD(AgentStats agentStat) {
                return (
                    agentStat.infectedTimestamp > timestamp - 24 * 60 / timeStepL && agentStat.infectedTimestamp <= timestamp);
            });
        stats.push_back(newInfected);
        if (print)
            std::cout << newInfected << "\t";

        //Number of people infected at least once
        unsigned infectionCount =
            thrust::count_if(agentStats.begin(), agentStats.end(), [] HD(AgentStats agentStat) {
                return agentStat.infectedCount>0;
            });
        stats.push_back(infectionCount);
        if (print)
            std::cout << infectionCount << "\t";

        //Number of reinfections so far
        unsigned reinfectionCount =
            thrust::transform_reduce(agentStats.begin(), agentStats.end(), 
                                     [] HD(AgentStats agentStat) {return unsigned(agentStat.infectedCount > 1 ? agentStat.infectedCount-1 : 0);},
                                     (unsigned)0, thrust::plus<unsigned>());
        stats.push_back(reinfectionCount);
        if (print)
            std::cout << reinfectionCount << "\t";

        //Cumulative number of boosters
        unsigned boosters =
            thrust::transform_reduce(agentStats.begin(), agentStats.end(), 
                                     [] HD(AgentStats agentStat) {return unsigned(agentStat.immunizationCount > 1 ? agentStat.immunizationCount-1 : 0);},
                                     (unsigned)0, thrust::plus<unsigned>());
        stats.push_back(boosters);
        if (print)
            std::cout << boosters << "\t";

        //Level of immunity in the population
        std::string out;
        float susceptib;
        for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++) {
            susceptib =
            thrust::transform_reduce(ppstates.begin(), ppstates.end(), 
                                    [variant] HD(PPState state) {return 1.0f-state.getSusceptible(variant);},
                                    0.0f, thrust::plus<float>());
            if (variant == 0) out = std::to_string(unsigned(susceptib));
            else out = out + "," + std::to_string(unsigned(susceptib));
            stats.push_back((unsigned)susceptib);
        }
        if (print)
            std::cout << out << "\t";

        //count number of health workers who are infected
        auto& quarantined = agents->quarantined;
        unsigned infectedHCWorker =
            thrust::count_if(thrust::make_zip_iterator(thrust::make_tuple(healthcareWorker.begin(),ppstates.begin(),quarantined.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(healthcareWorker.end(),ppstates.end(),quarantined.end())),
                            []HD(thrust::tuple<bool, PPState, bool> tup) {
                                return thrust::get<0>(tup) && (thrust::get<1>(tup).isInfected() || thrust::get<2>(tup));
                            });
        //count number of health workers exposed today
        unsigned exposedHCWorker =
            thrust::count_if(thrust::make_zip_iterator(thrust::make_tuple(healthcareWorker.begin(),agentStats.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(healthcareWorker.end(),agentStats.end())),
                            [timestamp, timeStepL]HD(thrust::tuple<bool, AgentStats> tup) {
                                return thrust::get<0>(tup) && 
                                    (thrust::get<1>(tup).infectedTimestamp > timestamp - 24 * 60 / timeStepL &&
                                     thrust::get<1>(tup).infectedTimestamp <= timestamp);
                            });
        stats.push_back((unsigned)(infectedHCWorker*100)/MAX(1,healthcareWorkerCount));
        stats.push_back((unsigned)exposedHCWorker);
        if (print)
            std::cout << (unsigned)(infectedHCWorker*100)/MAX(1,healthcareWorkerCount) << "\t";
        if (print)
            std::cout << (unsigned)exposedHCWorker << "\t";

        //Number of people infected with different variants
        unsigned countInf;
        for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++) {
            countInf =
            thrust::count_if(agentStats.begin(), agentStats.end(), 
                                    [variant] HD(AgentStats stats) {return ((1<<variant) & stats.variant) ? true : false;});
            if (variant == 0) out = std::to_string(unsigned(countInf));
            else out = out + "," + std::to_string(unsigned(countInf));
            stats.push_back((unsigned)countInf);
        }
        if (print)
            std::cout << out << "\t";

        //Number of people in hospital with active infection
        unsigned hospitalTypeLocal = hospitalType;
        unsigned infInHosp = thrust::count_if(thrust::make_zip_iterator(
            thrust::make_tuple(ppstates.begin(),thrust::make_permutation_iterator(locs->locType.begin(), agents->location.begin()))),
                                              thrust::make_zip_iterator(
            thrust::make_tuple(ppstates.end(),thrust::make_permutation_iterator(locs->locType.begin(), agents->location.end()))),
            [hospitalTypeLocal] HD (thrust::tuple<PPState, unsigned> tup) {return (thrust::get<0>(tup).isInfected() && thrust::get<1>(tup)==hospitalTypeLocal);});
        stats.push_back(infInHosp);
        if (print)
            std::cout << infInHosp << "\t";

        //Number of vaccinated but not previously infected
        unsigned vaccNotInf =
            thrust::count_if(agentStats.begin(), agentStats.end(), 
                                     [] HD(AgentStats agentStat) {return agentStat.infectedCount == 0 && agentStat.immunizationCount>0;});
        stats.push_back(vaccNotInf);
        if (print)
            std::cout << vaccNotInf;

        if (print)
            std::cout << '\n';
        return stats;
    }

    void flagHealthcareWorkers(){
        auto& ppstates = agents->PPValues;
        auto& agentStats = agents->agentStats;

        healthcareWorker.resize(ppstates.size());
        //Tools to determine if health worker
        auto* locationOffsetPtr = thrust::raw_pointer_cast(agents->locationOffset.data());
        auto* possibleTypesPtr = thrust::raw_pointer_cast(agents->possibleTypes.data());
        auto* locationTypePtr = thrust::raw_pointer_cast(locs->locType.data());
        auto* possibleLocationsPtr = thrust::raw_pointer_cast(agents->possibleLocations.data());
        auto healthworker = [locationOffsetPtr, possibleTypesPtr, possibleLocationsPtr, locationTypePtr] HD(
                                    unsigned id) -> bool {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
                if (possibleTypesPtr[idx] == 4
                    && (locationTypePtr[possibleLocationsPtr[idx]] == 12 || locationTypePtr[possibleLocationsPtr[idx]] == 14))
                    return true;
            }
            return false;
        };

        thrust::transform(thrust::make_counting_iterator(unsigned(0)),
                            thrust::make_counting_iterator(unsigned(agentStats.size())),
                            healthcareWorker.begin(), healthworker);

        healthcareWorkerCount = thrust::count(healthcareWorker.begin(),healthcareWorker.end(),true);
    }

    void setupHospitalizations(const cxxopts::ParseResult& result) {
        dueWithCOVID = result["dueWithCOVID"].as<unsigned>();
        std::string countStr = result["totalHospitalizations"].as<std::string>();
        std::ifstream t(countStr.c_str());
        std::string buffer;

        while(std::getline(t, buffer)) {
            if (buffer.length()==0 || buffer.at(0) == '#')
                continue;
            totalHospitalizations = splitStringInt(buffer, ',');
        }
    }

public:
    explicit Simulation(const cxxopts::ParseResult& result)
        : timeStep(result["deltat"].as<decltype(timeStep)>()),
          lengthOfSimulationWeeks(result["weeks"].as<decltype(lengthOfSimulationWeeks)>()),
          simTime(timeStep, 0, static_cast<Days>(result["startDay"].as<unsigned>()), result["startDate"].as<unsigned>()) {
        PROFILE_FUNCTION();
        outAgentStat = result["outAgentStat"].as<std::string>();
        enableOtherDisease = result["otherDisease"].as<int>();
        infectiousnessMultiplier = splitStringFloat(result["infectiousnessMultiplier"].as<std::string>(),',');
        diseaseProgressionScaling = splitStringFloat(result["diseaseProgressionScaling"].as<std::string>(),',');
        diseaseProgressionDeathScaling = splitStringFloat(result["diseaseProgressionDeathScaling"].as<std::string>(),',');
        diagnosticLevel = result["diags"].as<unsigned>();
        setupHospitalizations(result);
        #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
        if (omp_get_max_threads() == 1)
            Util::needAgentsSortedByLocation = 0;
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
            agents->initAgents(data.acquireAgents(),
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

    std::vector<unsigned> countVariantCases() {
        auto& ppstates = agents->PPValues;
        std::vector<unsigned> variantCounts;
        for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++) {
            variantCounts.push_back(thrust::count_if(ppstates.begin(), ppstates.end(), [variant] HD(PPState state) {
                return state.isInfected() && state.getVariant() == variant;
            }));
        }
        return variantCounts;
    }

    void processFlags(char **argv, int argc) {
        //Possible flags: "TPdef", "TP015", "TP035", "PLNONE", "PL0", "CFNONE","CF2000-0500", "SONONE", "SO12", "SO3", "QU0", "QU2", "QU3", "MA1.0", "MA0.8"
        for (int i = 0; i < argc; i++) {
            std::string flag = argv[i];
            std::string prefix = flag.substr(0, 2);

            if (prefix == "TP") {
                if (diagnosticLevel > 0) std::cout << "Testing policy " << flag << std::endl;
                if (flag == "TPdef") {
                    updateTestingProbs("0.00005,0.01,0.0005,0.0005,0.005,0.05");
                } else if (flag == "TP015") {
                    updateTestingProbs("0.00005,0.2,0.04,0.04,0.005,0.05");
                } else if (flag == "TP035") {
                    updateTestingProbs("0.00005,1.0,0.2,0.2,0.005,0.05");
                }
            } else if (prefix == "PL") {
                if (diagnosticLevel > 0) std::cout << "Location closures " << flag << std::endl;
                if (flag == "PLNONE") {
                    int fixListArr[4] = {5,6,22,44};
                    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                                             locs->locType.begin(), locs->states.begin(), locs->essential.begin())),
                            thrust::make_zip_iterator(
                                thrust::make_tuple(locs->locType.end(), locs->states.end(), locs->essential.end())),
                            [fixListArr] HD(
                                thrust::tuple<unsigned&, bool&, uint8_t&> tup) {
                                auto& type = thrust::get<0>(tup);
                                auto& isOpen = thrust::get<1>(tup);
                                auto& isEssential = thrust::get<2>(tup);
                                if (isEssential == 1) return;
                                for (unsigned i = 0; i < 4; i++)
                                    if (type == fixListArr[i])
                                        isOpen = true;
                            });
                } else if (flag == "PL0") {
                    int fixListArr[4] = {5,6,22,44};
                    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                                             locs->locType.begin(), locs->states.begin(), locs->essential.begin())),
                            thrust::make_zip_iterator(
                                thrust::make_tuple(locs->locType.end(), locs->states.end(), locs->essential.end())),
                            [fixListArr] HD(
                                thrust::tuple<unsigned&, bool&, uint8_t&> tup) {
                                auto& type = thrust::get<0>(tup);
                                auto& isOpen = thrust::get<1>(tup);
                                auto& isEssential = thrust::get<2>(tup);
                                if (isEssential == 1) return;
                                for (unsigned i = 0; i < 4; i++)
                                    if (type == fixListArr[i])
                                        isOpen = false;
                            });

                }
            } else if (prefix == "CF") {
                if (diagnosticLevel > 0) std::cout << "Curfew policy: " << flag << std::endl;
                if (flag == "CFNONE") {
                    toggleCurfew(false, 0, 0);
                } else if (flag == "CF2000-0500") {
                    toggleCurfew(true, 20*60, 5*60);
                }
            } else if (prefix == "SO") {
                if (diagnosticLevel > 0) std::cout << "School age restriction: " << flag << std::endl;
                if (flag == "SONONE") {
                    setSchoolAgeRestriction(99);
                } else if (flag == "SO12") {
                    setSchoolAgeRestriction(12);
                } else if (flag == "SO3") {
                    setSchoolAgeRestriction(3);
                }
            } else if (prefix == "QU") {
                if (diagnosticLevel > 0) std::cout << "Quarantine policy: " << flag << std::endl;
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
                        if (diagnosticLevel > 0) std::cout << "Masks at 100%" << std::endl;
                        double currentMaskValue_local = currentMaskValue;
                        int homeType_l = homeType;
                        thrust::for_each(
                            thrust::make_zip_iterator(thrust::make_tuple(locs->locType.begin(), locs->infectiousness.begin())),
                            thrust::make_zip_iterator(thrust::make_tuple(locs->locType.end(), locs->infectiousness.end())),
                            [currentMaskValue_local, homeType_l] HD(
                                thrust::tuple<unsigned&, float&> tup) {
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
                        if (diagnosticLevel > 0) std::cout << "Masks at 80%" << std::endl;
                        double currentMaskValue_local = currentMaskValue;
                        int homeType_l = homeType;
                        thrust::for_each(
                            thrust::make_zip_iterator(thrust::make_tuple(locs->locType.begin(), locs->infectiousness.begin())),
                            thrust::make_zip_iterator(thrust::make_tuple(locs->locType.end(), locs->infectiousness.end())),
                            [currentMaskValue_local, homeType_l] HD(
                                thrust::tuple<unsigned&, float&> tup) {
                                auto& type = thrust::get<0>(tup);
                                auto& infectiousness = thrust::get<1>(tup);
                                if (type != homeType_l) {
                                    infectiousness = infectiousness * currentMaskValue_local;
                                }
                            });
                        currentMaskValue = 0.8;
                    }
                }
            }
        }
    }

    void runSimulation() {
        std::vector<unsigned> variantCounts;
        if (!succesfullyInitialized) { return; }
        PROFILE_FUNCTION();
        const Timehandler endOfSimulation(timeStep, lengthOfSimulationWeeks, Days::MONDAY);
        while (simTime < endOfSimulation) {
            // std::cout << simTime.getTimestamp() << std::endl;
            if (simTime.isMidnight()) {
                BEGIN_PROFILING("midnight")
                if (simTime.getTimestamp() > 0) TestingPolicy<Simulation>::performTests(simTime, timeStep);
                if (simTime.getTimestamp() > 0) updateAgents(simTime);// No disease progression at launch
                immunization->update(simTime, timeStep);
                if (enableOtherDisease) otherDisease(simTime, timeStep);
                auto stats = refreshAndPrintStatistics(simTime);
                ClosurePolicy<Simulation>::midnight(simTime, timeStep, stats);
                MovementPolicy<Simulation>::planLocations(simTime, timeStep);
                variantCounts = countVariantCases(); //TODO: we do this multiple times
                END_PROFILING("midnight")
            }
            MovementPolicy<Simulation>::movement(simTime, timeStep);
            ClosurePolicy<Simulation>::step(simTime, timeStep);
            for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++)
                if (variantCounts[variant]>0) InfectionPolicy<Simulation>::infectionsAtLocations(simTime, timeStep, variant);
            ++simTime;
        }
        agents->printAgentStatJSON(outAgentStat);
        InfectionPolicy<Simulation>::finalize();
    }

    std::vector<unsigned> runForDay(int argc, char **args) {
        std::vector<unsigned> variantCounts;
        
        PROFILE_FUNCTION();
        processFlags(args, argc);
        auto stats = refreshAndPrintStatistics(simTime,false);
        ClosurePolicy<Simulation>::midnight(simTime, timeStep, stats);
        MovementPolicy<Simulation>::planLocations(simTime, timeStep);
        variantCounts = countVariantCases(); //TODO: we do this multiple times

        unsigned stepsPerDay = simTime.getStepsPerDay();
        for (int i = 0; i < stepsPerDay; i++) {
            MovementPolicy<Simulation>::movement(simTime, timeStep);
            ClosurePolicy<Simulation>::step(simTime, timeStep);
            for (int variant = 0; variant < infectiousnessMultiplier.size(); variant++)
                if (variantCounts[variant]>0) InfectionPolicy<Simulation>::infectionsAtLocations(simTime, timeStep, variant);
            ++simTime;
            
        }
        BEGIN_PROFILING("midnight")
        if (simTime.getTimestamp() > 0) TestingPolicy<Simulation>::performTests(simTime, timeStep);
        if (simTime.getTimestamp() > 0) updateAgents(simTime);// No disease progression at launch
        immunization->update(simTime, timeStep);
        if (enableOtherDisease) otherDisease(simTime, timeStep);
        stats = refreshAndPrintStatistics(simTime);
        END_PROFILING("midnight")

        return stats;
    }
    
    void finalize() {
        agents->printAgentStatJSON(outAgentStat);
        InfectionPolicy<Simulation>::finalize();
    }

    void toggleCurfew(bool enable, unsigned curfewBegin, unsigned curfewEnd) {
        MovementPolicy<Simulation>::enableCurfew = enable;
        MovementPolicy<Simulation>::curfewBegin = curfewBegin;
        MovementPolicy<Simulation>::curfewEnd = curfewEnd;
        MovementPolicy<Simulation>::curfewTimeConverted = false;
    }
    void setSchoolAgeRestriction(unsigned limit) { MovementPolicy<Simulation>::schoolAgeRestriction = limit; }
    void toggleHolidayMode(bool enable) { MovementPolicy<Simulation>::holidayModeActive = enable; }
    void toggleQuarantineImmune(bool enable) { MovementPolicy<Simulation>::quarantineImmuneActive = enable; }
    void toggleLockdownNonvacc(bool enable) { MovementPolicy<Simulation>::lockdownNonvaccActive = enable; }
    void quarantinePolicy(unsigned newQP) { MovementPolicy<Simulation>::quarantinePolicy = newQP; }
    void updateTestingProbs(const std::string &probs) {TestingPolicy<Simulation>::updateTestingProbs(probs);};
    Timehandler& getSimTime() { return simTime; }
};
