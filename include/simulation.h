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

    void otherDisease(Timehandler& simTime, unsigned timeStep);

    void updateAgents(Timehandler& simTime);

    std::vector<unsigned> refreshAndPrintStatistics(Timehandler& simTime, bool print = true);

    void flagHealthcareWorkers();

    void setupHospitalizations(const cxxopts::ParseResult& result);

public:
    explicit Simulation(const cxxopts::ParseResult& result);

    std::vector<unsigned> countVariantCases();
    void processFlags(char **argv, int argc);
    void runSimulation();
    std::vector<unsigned> runForDay(int argc, char **args);
    void finalize();

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
