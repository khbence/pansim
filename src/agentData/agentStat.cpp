#include "agentStats.h"

std::ostream& operator<<(std::ostream& os, const AgentStats& s) {
    if (s.infectedTimestamp == std::numeric_limits<decltype(s.infectedTimestamp)>::max()) {
        return os;
    }
    os << "Infected at " << s.infectedTimestamp << " with variant " << s.variant << " location " << s.infectedLocation
       << " diagnosed: " << s.diagnosedTimestamp << " last quarantined: " << s.quarantinedTimestamp
       << " - " << s.quarantinedUntilTimestamp << " total days in quarantine " << s.daysInQuarantine;
    os << " worst state " << static_cast<unsigned>(s.worstState)
       << " between: " << s.worstStateTimestamp << "-" << s.worstStateEndTimestamp;
    os << " hospitalized " << static_cast<unsigned>(s.hospitalizedTimestamp) << "-" 
       << s.hospitalizedUntilTimestamp << " immunized " << static_cast<unsigned>(s.immunizationTimestamp) << "\n";
    return os;
}