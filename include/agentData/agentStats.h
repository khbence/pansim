#pragma once
#include <limits>
#include <iostream>

class AgentStats {
public:
    unsigned infectedTimestamp = std::numeric_limits<unsigned>::max();
    unsigned infectedLocation = 0;
    unsigned worstStateTimestamp = 0;
    unsigned worstStateEndTimestamp = 0;
    unsigned diagnosedTimestamp = 0;
    unsigned quarantinedTimestamp = 0;
    unsigned quarantinedUntilTimestamp = 0;
    unsigned daysInQuarantine = 0;
    unsigned hospitalizedTimestamp = 0;
    unsigned hospitalizedUntilTimestamp = 0;
    unsigned immunizationTimestamp = 0;
    unsigned recoveredTimestamp = std::numeric_limits<unsigned>::max();
    uint8_t immunizationCount = 0;
    unsigned variant = 0;
    uint8_t infectedCount = 0;
    char worstState = 0;
    friend std::ostream& operator<<(std::ostream& os, const AgentStats& s);
};