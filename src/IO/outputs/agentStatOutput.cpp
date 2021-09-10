#include "agentStatOutput.h"
#include <limits>
#include "dynamicPPState.h"

AgentStatOutput::AgentStatOutput(const thrust::host_vector<AgentStats>& data) {
    rapidjson::Value stats(rapidjson::kArrayType);
    const auto& names = DynamicPPState::getStateNames();
    for (unsigned idx = 0; idx < data.size(); idx++) {
        auto &e = data[idx];
        // if (e.infectedTimestamp != std::numeric_limits<decltype(e.infectedTimestamp)>::max() || e.worstState != 0) {
            rapidjson::Value currentAgent(rapidjson::kObjectType);
            currentAgent.AddMember("ID", idx, allocator);
            currentAgent.AddMember("infectionTime", e.infectedTimestamp, allocator);
            currentAgent.AddMember("infectedCount", e.infectedCount, allocator);
            currentAgent.AddMember("variant", e.variant, allocator);
            currentAgent.AddMember("InfectionLoc", e.infectedLocation, allocator);
            currentAgent.AddMember("diagnosisTime", e.diagnosedTimestamp, allocator);
            currentAgent.AddMember("quarantinedTime", e.quarantinedTimestamp, allocator);
            currentAgent.AddMember("quarantinedUntilTime", e.quarantinedUntilTimestamp, allocator);
            currentAgent.AddMember("daysInQuarantine", e.daysInQuarantine, allocator);
            currentAgent.AddMember("hospitalizedTime", e.hospitalizedTimestamp, allocator);
            currentAgent.AddMember("hospitalizedUntilTime", e.hospitalizedUntilTimestamp, allocator);
            currentAgent.AddMember("immunizationTimestamp", e.immunizationTimestamp, allocator);
            rapidjson::Value worst(rapidjson::kObjectType);
            worst.AddMember("name", stringToObject(names[e.worstState]), allocator);
            worst.AddMember("begin", e.worstStateTimestamp, allocator);
            if (e.worstStateEndTimestamp == 0) {
                worst.AddMember("end", -1, allocator);
            } else {
                worst.AddMember("end", e.worstStateEndTimestamp, allocator);
            }
            currentAgent.AddMember("worstState", worst, allocator);
            stats.PushBack(currentAgent, allocator);
        // }
    }
    d.AddMember("Statistics", stats, allocator);
}