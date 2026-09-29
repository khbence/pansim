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

    thrust::tuple<unsigned, unsigned, unsigned> getQuarantineStats(unsigned timestamp) {
        thrust::tuple<unsigned, unsigned, unsigned> res = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(agentStats.begin(), PPValues.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(agentStats.end(), PPValues.end())),
            [timestamp] HD(thrust::tuple<AgentStats, PPState> tup) {
                AgentStats& stat = thrust::get<0>(tup);
                PPState& ppstate = thrust::get<1>(tup);
                unsigned isQuarantined =
                    unsigned(stat.quarantinedTimestamp <= timestamp && stat.quarantinedUntilTimestamp > timestamp);
                // Is currently quarantined
                // If quarantined, is infected?
                // Not quarantined, but infected
                return thrust::make_tuple(isQuarantined,
                    unsigned(isQuarantined && ppstate.isInfected()),
                    unsigned(!isQuarantined && ppstate.isInfected()));
            },
            thrust::make_tuple(unsigned(0), unsigned(0), unsigned(0)),
            [] HD(thrust::tuple<unsigned, unsigned, unsigned> a, thrust::tuple<unsigned, unsigned, unsigned> b) {
                return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                    thrust::get<1>(a) + thrust::get<1>(b),
                    thrust::get<2>(a) + thrust::get<2>(b));
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

    [[nodiscard]] std::map<unsigned, unsigned> initAgentTypes(const parser::AgentTypes& inputData);

    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("disableTourists",
            "enable or disable tourists",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(1))));
    }

    void initializeArgs(const cxxopts::ParseResult& result);

    void initAgents(parser::Agents& inputData,
        const std::map<std::string, unsigned>& locMap,
        const std::map<unsigned, unsigned>& typeMap,
        const std::map<unsigned, std::vector<unsigned>>& agentTypeLocType,
        const std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>& progressionMatrices,
        const parser::LocationTypes& locationTypes);

    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    auto getPPState(unsigned i) { return PPValues[i]; }

    void printAgentStatJSON(const std::string& fileName);
};
