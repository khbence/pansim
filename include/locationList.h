#pragma once
#include <vector>
#include "globalStates.h"
#include "agent.h"
#include <cmath>
#include <algorithm>
#include <iterator>
#include <random>
#include "randomGenerator.h"
#include "statistics.h"
#include "datatypes.h"
#include "timing.h"
#include "util.h"
#include <string>
#include "locationTypesFormat.h"
#include <map>
#include "locationsFormat.h"
#include "customExceptions.h"
#include "timeHandler.h"
#include <cxxopts.hpp>

template<typename SimulationType>
class LocationsList {
    using AgentType = Agent<typename SimulationType::AgentListType>;

    using PositionType = typename SimulationType::PositionType_t;
    using TypeOfLocation = typename SimulationType::TypeOfLocation_t;

    Statistic<typename SimulationType::PPState_t, AgentType> globalStats;

    LocationsList() = default;

    void reserve(std::size_t s) {
        position.reserve(s);
        locType.reserve(s);
        areas.reserve(s);
        capacity.reserve(s);
        states.reserve(s);
        quarantineUntil.reserve(s);
        closedUntil.reserve(s);
        essential.reserve(s);
    }

public:
    // the following vectors are the input data for locations in separated
    // vectors
    thrust::device_vector<TypeOfLocation> locType;
    thrust::device_vector<PositionType> position;
    thrust::device_vector<float> infectiousness;
    thrust::device_vector<float> eventBoost;
    thrust::device_vector<unsigned> areas;
    thrust::device_vector<unsigned> capacity;
    thrust::device_vector<bool> states;// Closed/open or ON/OFF
    thrust::device_vector<unsigned> quarantineUntil;
    thrust::device_vector<unsigned> closedUntil;

    thrust::device_vector<unsigned> schools;
    thrust::device_vector<unsigned> classrooms;
    thrust::device_vector<unsigned> classroomOffsets;
    thrust::device_vector<uint8_t> essential;


    // indices of agents sorted by location, and sorted by agent index
    thrust::device_vector<unsigned> locationAgentList;
    // indices of locations of the agents sorted
    // by location, and sorted by agent index
    thrust::device_vector<unsigned> locationIdsOfAgents;
    // into locationAgentList
    thrust::device_vector<unsigned> locationListOffsets;

    std::map<unsigned, std::string> generalLocationTypes;

    unsigned tracked;
    unsigned venueProfile = 0;
    double venueStrength = 1.0;
    double communityScale = 1.0;
    double eventRate = 0.0;
    double eventFactor = 20.0;
    double homeVenueFactor = -1.0;
    int lastEventDay = -1;
    std::vector<char> eventTypeAllowed = std::vector<char>(64, 0);
    thrust::device_vector<uint8_t> eventEligible;

    [[nodiscard]] static LocationsList* getInstance() {
        static LocationsList instance;
        return &instance;
    }

    void initLocationTypes(const parser::LocationTypes& inputData);

    void initializeArgs(const cxxopts::ParseResult& result);

    [[nodiscard]] std::pair<unsigned, std::map<std::string, unsigned>> initLocations(
        const parser::Locations& inputData,
        const parser::LocationTypes& locTypes);

    void initialize();
    void applyVenueAmplification();
    void updateDailyEvents(unsigned day);

    // TODO optimise randoms for performance
    static void infectAgents(thrust::device_vector<float>& infectionRatioAtLocations,
        thrust::device_vector<unsigned>& agentLocations,
        thrust::device_vector<bool>& infectionAtLocation,
        thrust::device_vector<unsigned>& newlyInfectedAgents,
        bool flagInfectionsAtLocation,
        Timehandler& simTime,
        uint8_t variant,
        double sheddingShape,
        const float* traitInfectiousness = nullptr,
        const float* traitSusceptibility = nullptr,
        bool peakMode = false) {
        //        PROFILE_FUNCTION();
        auto& ppstates = SimulationType::AgentListType::getInstance()->PPValues;
        auto& agentStats = SimulationType::AgentListType::getInstance()->agentStats;
        unsigned timestamp = simTime.getTimestamp();
        unsigned tracked2 = getInstance()->tracked;
        unsigned hour = simTime.getMinutes() / 60;
        // DEBUG unsigned count1 =
        // thrust::count_if(ppstates.begin(),ppstates.end(), [](auto &ppstate)
        // {return ppstate.getSIRD() == states::SIRD::I;}); DESC: for (int i =
        // 0; i < number_of_agents; i++) {ppstate = ppstates[i]; infectionRatio
        // = infectionRatioAtLocations[agentLocations[i]];...}
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                             thrust::make_permutation_iterator(infectionRatioAtLocations.begin(), agentLocations.begin()),
                             thrust::make_permutation_iterator(getInstance()->locType.begin(), agentLocations.begin()),
                             agentStats.begin(),
                             agentLocations.begin(),
                             thrust::make_counting_iterator<unsigned>(0),
                             thrust::make_permutation_iterator(infectionAtLocation.begin(), agentLocations.begin()),
                             newlyInfectedAgents.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                thrust::make_permutation_iterator(infectionRatioAtLocations.begin(), agentLocations.end()),
                thrust::make_permutation_iterator(getInstance()->locType.begin(), agentLocations.end()),
                agentStats.end(),
                agentLocations.end(),
                thrust::make_counting_iterator<unsigned>(0) + ppstates.size(),
                thrust::make_permutation_iterator(infectionAtLocation.begin(), agentLocations.end()),
                newlyInfectedAgents.begin() + ppstates.size())),
            [timestamp, tracked2, flagInfectionsAtLocation, variant, sheddingShape, traitInfectiousness, traitSusceptibility, peakMode] HD(thrust::tuple<typename SimulationType::PPState_t&,
                float&,
                TypeOfLocation&,
                AgentStats&,
                unsigned&,
                unsigned,
                bool&,
                unsigned&> tuple) {
                auto& ppstate = thrust::get<0>(tuple);
                float& infectionRatio = thrust::get<1>(tuple);
                TypeOfLocation& locType = thrust::get<2>(tuple);
                auto& agentStat = thrust::get<3>(tuple);
                unsigned& agentLocation = thrust::get<4>(tuple);
                unsigned agentID = thrust::get<5>(tuple);
                bool& infectionAtLocation = thrust::get<6>(tuple);
                unsigned& newlyInfectedAgent = thrust::get<7>(tuple);
                float infectionProbability = infectionRatio * ppstate.getSusceptible(variant);
                if (traitSusceptibility != nullptr && ppstate.getSusceptible(variant) > 0.0f) {
                    float weight = traitSusceptibility[agentID];
                    if (weight != 1.0f) infectionProbability = -expm1f(weight * log1pf(-infectionProbability));
                }
                if (ppstate.getSusceptible(variant) > 0.0f
                    && RandomGenerator::randomUnit() < infectionProbability) {
                    ppstate.gotInfected(variant);
                    float shed;
                    if (traitInfectiousness != nullptr) {
                        shed = traitInfectiousness[agentID];
                    } else {
                        shed = sheddingShape > 0.0 ? static_cast<float>(RandomGenerator::gammaMeanOne(sheddingShape)) : 1.0f;
                    }
                    ppstate.setShedMultiplier(peakMode ? 1.0f : shed);
                    agentStat.shedMultiplier = shed;
                    agentStat.symptomOnsetTimestamp = 0xffffffffu;
                    agentStat.infectedTimestamp = timestamp;
                    agentStat.infectedCount++;
                    agentStat.infectedLocation = agentLocation;
                    agentStat.worstState = ppstate.getStateIdx();
                    agentStat.worstStateTimestamp = timestamp;
                    agentStat.variant |= 1<<variant;
                    if (flagInfectionsAtLocation) {
                        infectionAtLocation = true;
                        newlyInfectedAgent = 1;
                    }
                    if (agentID == tracked2) {
                        printf("Agent %d got infected with variant %d at location %d of type %d at timestamp %d\n",
                            agentID,
                            variant,
                            agentLocation,
                            locType,
                            timestamp);
                    }
                }
            });
        // DEBUG unsigned count2 =
        // thrust::count_if(ppstates.begin(),ppstates.end(), [](auto &ppstate)
        // {return ppstate.getSIRD() == states::SIRD::I;}); DEBUG std::cout <<
        // count1 << " " << count2 << std::endl;
    }

    const std::vector<unsigned>& refreshAndGetStatistic();
};
