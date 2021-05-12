#pragma once
#include "agentsList.h"
#include "globalStates.h"

template<typename AgentListType>
class Agent {
    static AgentListType* agentList;

public:
    unsigned id;
    using AgentListType_t = AgentListType;
    explicit HD Agent() : id(0) {}
    explicit HD Agent(unsigned id_p) : id(id_p) {}
    //[[nodiscard]] states::SIRD getSIRDState() const { return
    // agentList->PPValues[id].getSIRD(); }
    //[[nodiscard]] auto& getPPState() { return agentList->PPValues[id]; }
    void gotInfected() { agentList->PPValues[id].gotInfected(); }
    void progressDisease(float additionalFactor = 1.0) {
        float scalingFactor = additionalFactor * agentList->agentMetaData[id].getScalingSymptoms();
        agentList->PPValues[id].update(scalingFactor);
    }
};

template<typename AgentListType>
AgentListType* Agent<AgentListType>::agentList = AgentListType::getInstance();
