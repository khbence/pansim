#pragma once
#include "JSONWriter.h"
#include "datatypes.h"
#include "agentStats.h"

class AgentStatOutput : public JSONWriter {
public:
    explicit AgentStatOutput(const thrust::host_vector<AgentStats>& data);
};