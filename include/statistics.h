#pragma once
#include "datatypes.h"
#include "agent.h"
#include "globalStates.h"
#include <array>
#include <algorithm>
#include "timing.h"

template<typename PPStateType, typename AgentType>
class Statistic {
    std::vector<unsigned> states;
    // we can store here nice combined stats, if we don't wanna calculate them
    // all the time

public:
    void refreshStatisticNewAgent(const unsigned& a) {
        typename AgentType::AgentListType_t::PPState_t state =
            AgentType::AgentListType_t::getInstance()->PPValues[a];
        if (states.size() != PPStateType::getNumberOfStates())
            states.resize(PPStateType::getNumberOfStates());
        ++states[state.getStateIdx()];
    }

    void refreshStatisticRemoveAgent(const unsigned& a) {
        typename AgentType::AgentListType_t::PPState_t state =
            AgentType::AgentListType_t::getInstance()->PPValues[a];
        if (states.size() != PPStateType::getNumberOfStates())
            states.resize(PPStateType::getNumberOfStates());
        --states[state.getStateIdx()];
    }

    const decltype(states)& refreshandGetAfterMidnight(const std::pair<unsigned, unsigned>& agents,
        const thrust::device_vector<unsigned>& locationAgentList) {
        //PROFILE_FUNCTION();
        // Extract Idxs
        unsigned numAgentsAtLocation = agents.second - agents.first;
        thrust::device_vector<char> idxs(numAgentsAtLocation);
        auto ppstates = AgentType::AgentListType_t::getInstance()->PPValues;
        // DEBUG thrust::copy(locationAgentList.begin()+agents.first,
        // locationAgentList.begin()+agents.second,
        // std::ostream_iterator<int>(std::cout, "
        // ")); std::cout << std::endl; DESC: for (unsigned i = agents.first; i
        // < agents.second; i++) {ppstate = ppstates[locationAgentList[i]];}
        thrust::transform(thrust::make_permutation_iterator(
                              ppstates.begin(), locationAgentList.begin() + agents.first),
            thrust::make_permutation_iterator(
                ppstates.begin(), locationAgentList.begin() + agents.second),
            idxs.begin(),
            [] HD(PPStateType & ppstate) { return ppstate.getStateIdx(); });
        // Sort them
        thrust::sort(idxs.begin(), idxs.end());
        thrust::host_vector<int> h_idxs(idxs);
        // DEBUG thrust::copy(h_idxs.begin(), h_idxs.end(),
        // std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;

        thrust::device_vector<char> d_states(PPStateType::getNumberOfStates());
        thrust::device_vector<unsigned int> offsets(PPStateType::getNumberOfStates());
        thrust::sequence(d_states.begin(), d_states.end());// 0,1,...
        thrust::lower_bound(
            idxs.begin(), idxs.end(), d_states.begin(), d_states.end(), offsets.begin());
        thrust::host_vector<unsigned int> h_offsets(offsets);
        if (states.size() != PPStateType::getNumberOfStates())
            states.resize(PPStateType::getNumberOfStates());
        for (int i = 0; i < offsets.size() - 1; i++) {
            states[i] = h_offsets[i + 1] - h_offsets[i];
        }
        states.back() = numAgentsAtLocation - h_offsets.back();
        return states;
    }
};