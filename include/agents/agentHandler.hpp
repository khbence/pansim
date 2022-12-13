#pragma once
#include <thrust/device_vector.h>
#include "PPStateTypes.hpp"

class AgentHandler {
    thrust::device_vector<PPState> PPValues;
    thrust::device_vector<AgentMeta> agentMetaData;
    // id in the array of the progression matrices
    thrust::device_vector<bool> diagnosed;
    thrust::device_vector<unsigned> location;
    thrust::device_vector<unsigned> types;
    thrust::device_vector<AgentStats> agentStats;
    thrust::device_vector<bool> quarantined;
    thrust::device_vector<bool> stayedHome;

public:
};