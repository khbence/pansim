#pragma once
#include <vector>
#include "datatypes.h"
#include <string>
#include "agentType.h"
#include <map>
#include "parametersFormat.h"
#include "agentTypesFormat.h"
#include "customExceptions.h"
#include "globalStates.h"
#include "timeHandler.h"
#include <iterator>
#include "agentsFormat.h"
#include "agentMeta.h"
#include "agentStats.h"
#include "agentStatOutput.h"
#include "progressionMatrixFormat.h"
#include "dataProvider.h"
#include "progressionType.h"

template<typename T>
class Agent;

template<typename PPState, typename AgentMeta, typename Location>
class AgentList {
    AgentList() = default;

    void reserve(std::size_t s) {
        PPValues.reserve(s);
        agentMetaData.reserve(s);
        diagnosed.reserve(s);
        quarantined.reserve(s);
        location.reserve(s);
        agents.reserve(s);
        stayedHome.reserve(s);
    }

public:
    AgentTypeList agentTypes;
    thrust::device_vector<PPState> PPValues;
    thrust::device_vector<AgentMeta> agentMetaData;
    // id in the array of the progression matrices
    thrust::device_vector<bool> diagnosed;
    thrust::device_vector<unsigned> location;
    thrust::device_vector<unsigned> types;
    thrust::device_vector<AgentStats> agentStats;
    thrust::device_vector<bool> quarantined;
    thrust::device_vector<bool> stayedHome;

    thrust::device_vector<unsigned long> locationOffset;
    // longer, every agents' every locations, indexed by the offset
    thrust::device_vector<unsigned> possibleLocations;
    thrust::device_vector<unsigned> possibleTypes;

    thrust::tuple<unsigned,unsigned,unsigned> getQuarantineStats(unsigned timestamp) {
        thrust::tuple<unsigned,unsigned,unsigned> res =
            thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(agentStats.begin(), PPValues.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(agentStats.end(), PPValues.end())),
                                 [timestamp]HD(thrust::tuple<AgentStats,PPState> tup) {
                                     AgentStats &stat = thrust::get<0>(tup);
                                     PPState &ppstate = thrust::get<1>(tup);
                                     unsigned isQuarantined = unsigned(stat.quarantinedTimestamp <= timestamp && stat.quarantinedUntilTimestamp > timestamp);
                                     //Is currently quarantined
                                     //If quarantined, is infected?
                                     //Not quarantined, but infected
                                     return thrust::make_tuple(
                                        isQuarantined, 
                                        unsigned(isQuarantined && ppstate.isInfected()),
                                        unsigned(!isQuarantined && ppstate.isInfected()));
                                 },
                                 thrust::make_tuple(unsigned(0),unsigned(0),unsigned(0)),
                                 []HD(thrust::tuple<unsigned,unsigned,unsigned> a, thrust::tuple<unsigned,unsigned,unsigned> b) {
                                     return thrust::make_tuple(thrust::get<0>(a)+thrust::get<0>(b),
                                                               thrust::get<1>(a)+thrust::get<1>(b),
                                                               thrust::get<2>(a)+thrust::get<2>(b));
                                 });
        return res;
    }

    using PPState_t = PPState;

    friend class Agent<AgentList>;

    thrust::device_vector<Agent<AgentList>> agents;

    unsigned disableTourists;
    unsigned diagnosticLevel;
    unsigned quarantinePolicy = 0;
    unsigned quarantineLength = 0;
    unsigned timeStep;

    void initAgentMeta(const parser::Parameters& data) { AgentMeta::initData(data); }

    [[nodiscard]] std::map<unsigned, unsigned> initAgentTypes(const parser::AgentTypes& inputData) {
        // For the runtime performance, it would be better, that the IDs of the
        // agent types would be the same as their indexes, but we can not ensure
        // it in the input file, so I create this mapping, that will be used by
        // the agents when I fill them up. Use it only during initialization ID
        // from files -> index in vectors
        std::map<unsigned, unsigned> agentTypeIDMapping;
        agentTypes = AgentTypeList(inputData.types.size());
        // agent types
        unsigned idx = 0;
        for (auto& type : inputData.types) {
            agentTypeIDMapping.emplace(type.ID, idx);
            for (const auto& sch : type.schedulesUnique) {
                auto wb = states::parseWBState(sch.WB);
                auto days = Timehandler::parseDays(sch.dayType);

                std::vector<AgentTypeList::Event> events;
                events.reserve(sch.schedule.size());
                for (const auto& e : sch.schedule) { events.emplace_back(e); }
                for (auto day : days) {
                    agentTypes.addSchedule(idx, std::make_pair(wb, day), events);
                }
            }
            ++idx;
        }

        return agentTypeIDMapping;
    }

    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("disableTourists",
            "enable or disable tourists",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(1))));
    }

    void initializeArgs(const cxxopts::ParseResult& result) {
        disableTourists = result["disableTourists"].as<unsigned>();
        diagnosticLevel = result["diags"].as<unsigned>();
        try {
            quarantinePolicy = result["quarantinePolicy"].as<unsigned>();
            quarantineLength = result["quarantineLength"].as<unsigned>();
        } catch (std::exception &e) {}
        timeStep = result["deltat"].as<unsigned>();

    }

    void initAgents(parser::Agents& inputData,
        const std::map<std::string, unsigned>& locMap,
        const std::map<unsigned, unsigned>& typeMap,
        const std::map<unsigned, std::vector<unsigned>>& agentTypeLocType,
        const std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>&
            progressionMatrices,
        const parser::LocationTypes& locationTypes) {
        auto n = inputData.people.size();
        reserve(n);

        unsigned homeType = locationTypes.home;

        thrust::host_vector<PPState> PPValues_h;
        thrust::host_vector<AgentStats> agentStats_h;
        thrust::host_vector<AgentMeta> agentMetaData_h;
        thrust::host_vector<bool> diagnosed_h;
        thrust::host_vector<bool> quarantined_h;
        thrust::host_vector<bool> stayedHome_h;
        thrust::host_vector<unsigned> location_h;
        thrust::host_vector<unsigned> types_h;
        thrust::host_vector<Agent<AgentList>> agents_h;

        thrust::host_vector<unsigned long> locationOffset_h;
        // longer, every agents' every locations, indexed by the offset
        thrust::host_vector<unsigned> possibleLocations_h;
        thrust::host_vector<unsigned> possibleTypes_h;

        PPValues_h.reserve(n);
        agentMetaData_h.reserve(n);
        diagnosed_h.reserve(n);
        quarantined_h.reserve(n);
        location_h.reserve(n);
        types_h.reserve(n);
        agents_h.reserve(n);
        locationOffset_h.reserve(n + 1);
        locationOffset_h.push_back(0);
        agentStats_h.reserve(n);

        for (auto& person : inputData.people) {
            if (disableTourists && person.typeID == 9) continue;
            auto tmp = std::make_pair(
                static_cast<unsigned>(person.age), static_cast<std::string>(person.preCond));
            auto it = progressionMatrices.find(tmp);
            PPValues_h.push_back(PPState(person.state, it->second.second));
            AgentStats stat;
            if (PPValues_h.back().isInfected() > 0) {// Is infected at the beginning
                stat.infectedTimestamp = 0;
                stat.worstState = PPValues_h.back().getStateIdx();
                stat.worstStateTimestamp = 0;
            }
            if (person.diagnosed) {
                stat.diagnosedTimestamp = 1; //0 means was not diagnosed
                if (quarantinePolicy>0) {
                    stat.quarantinedTimestamp = 0;
                    stat.quarantinedUntilTimestamp = quarantineLength * 24 * 60 / timeStep;
                    stat.daysInQuarantine = quarantineLength;
                }
            }
            agentStats_h.push_back(stat);

            if (person.sex.size() != 1) { throw IOAgents::InvalidGender(person.sex); }
            agentMetaData_h.push_back(
                BasicAgentMeta(person.sex.front(), person.age, person.preCond));

            // I don't know if we should put any data about it in the input
            diagnosed_h.push_back(person.diagnosed);
            quarantined_h.push_back(person.diagnosed && quarantinePolicy>0);
            stayedHome_h.push_back(true);

            agents_h.push_back(Agent<AgentList>{ static_cast<unsigned>(agents.size()) });

            // agentType
            auto itType = typeMap.find(person.typeID);
            if (itType == typeMap.end()) { throw IOAgents::InvalidAgentType(person.typeID); }
            types_h.push_back(itType->second);

            // locations
            const auto& requestedLocs = agentTypeLocType.find(person.typeID)->second;
            std::vector<bool> hasThatLocType(requestedLocs.size(), false);
            std::vector<unsigned> locs;
            std::vector<unsigned> ts;// types
            locs.reserve(person.locations.size());
            ts.reserve(person.locations.size());
            std::sort(person.locations.begin(),
                person.locations.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.typeID < rhs.typeID; });
            for (const auto& l : person.locations) {
                auto itLoc = locMap.find(l.locID);
                if (itLoc == locMap.end()) { throw IOAgents::InvalidLocationID(l.locID); }
                locs.push_back(itLoc->second);
                ts.push_back(l.typeID);

                auto it = std::find(requestedLocs.begin(), requestedLocs.end(), l.typeID);
                if (it == requestedLocs.end()) {
                    //throw IOAgents::UnnecessaryLocType(
                    //    agents_h.size() - 1, person.typeID, l.typeID);
                } else 
                    hasThatLocType[std::distance(requestedLocs.begin(), it)] = true;
            }
            try {
            if (std::any_of(
                    hasThatLocType.begin(), hasThatLocType.end(), [](bool v) { return !v; })) {
                std::string missingTypes;
                for (unsigned idx = 0; idx < hasThatLocType.size(); ++idx) {
                    if (!hasThatLocType[idx]) {
                        missingTypes += std::to_string(requestedLocs[idx]) + ", ";
                    }
                }
                missingTypes.pop_back();
                missingTypes.pop_back();
                throw IOAgents::MissingLocationType(agents_h.size() - 1, types_h[types_h.size()-1]+1, std::move(missingTypes));
            }
            } catch (IOAgents::MissingLocationType& e) {
                std::cout << e.what() << std::endl;   
            }

            possibleLocations_h.insert(possibleLocations_h.end(), locs.begin(), locs.end());
            possibleTypes_h.insert(possibleTypes_h.end(), ts.begin(), ts.end());
            locationOffset_h.push_back(locationOffset_h.back() + locs.size());

            // First, put them in their homes if there is any
            auto it2 = std::find(ts.begin(), ts.end(), homeType);
            if (it2 == ts.end())
                location_h.push_back(0);
            else
                location_h.push_back(locs[std::distance(ts.begin(), it2)]);

        }

        PPValues = PPValues_h;
        agentMetaData = agentMetaData_h;
        diagnosed = diagnosed_h;
        quarantined = quarantined_h;
        location = location_h;
        types = types_h;
        agents = agents_h;
        locationOffset = locationOffset_h;
        possibleLocations = possibleLocations_h;
        possibleTypes = possibleTypes_h;
        agentStats = agentStats_h;
        stayedHome = stayedHome_h;
    }

    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    PPState& getPPState(unsigned i) { return PPValues[i]; }

    void printAgentStatJSON(const std::string& fileName) {
        AgentStatOutput writer{ agentStats };
        writer.writeFile(fileName);
        if (diagnosticLevel > 0) {
            //number of days in quarantine for ages 20-65
            unsigned days = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(agentStats.begin(), agentMetaData.begin())),
                                    thrust::make_zip_iterator(thrust::make_tuple(agentStats.end(), agentMetaData.end())),
                                    []HD(thrust::tuple<AgentStats, AgentMeta> tup) {
                                        auto& stat = thrust::get<0>(tup);
                                        auto& meta = thrust::get<1>(tup);
                                        return unsigned(stat.daysInQuarantine * (meta.getAge()>20 && meta.getAge()<=65));
                                    }, unsigned(0),thrust::plus<unsigned>());
            std::cout << "Total number of days in quarantine for agents 20<age<=65" << std::endl;
            std::cout << days << std::endl;
            //number of infections by location type
            unsigned cemeteryLocation = Location::getInstance()->locType.size()-1; //NOTE, we set the infected location for those who did NOT get infected to cemetery
            auto lambda = [cemeteryLocation]HD(AgentStats s) {return s.infectedTimestamp == std::numeric_limits<unsigned>::max() ? cemeteryLocation : s.infectedLocation;};
            thrust::device_vector<unsigned> agentInfectedLocType(agentStats.size());
            thrust::copy(thrust::make_permutation_iterator(Location::getInstance()->locType.begin(), thrust::make_transform_iterator(agentStats.begin(),lambda)),
                            thrust::make_permutation_iterator(Location::getInstance()->locType.begin(), thrust::make_transform_iterator(agentStats.end(),lambda)),
                            agentInfectedLocType.begin());
            thrust::sort(agentInfectedLocType.begin(),agentInfectedLocType.end());
            thrust::device_vector<unsigned> infectionsByLocTypeIDs(Location::getInstance()->generalLocationTypes.size()+2,0u);
            thrust::device_vector<unsigned> infectionsByLocTypeCount(Location::getInstance()->generalLocationTypes.size()+2,0u);
            thrust::reduce_by_key(agentInfectedLocType.begin(),agentInfectedLocType.end(),thrust::make_constant_iterator<unsigned>(1u),
                                infectionsByLocTypeIDs.begin(), infectionsByLocTypeCount.begin());
            //NOTE: since people who are not infected are assigned to cemetery locaiton, which is last, we don't print those
            thrust::copy(infectionsByLocTypeIDs.begin(), infectionsByLocTypeIDs.end(),
                    std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout << std::endl;
            thrust::copy(infectionsByLocTypeCount.begin(), infectionsByLocTypeCount.end(),
                    std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout << std::endl;
        }
    }
};
