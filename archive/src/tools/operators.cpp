#include "operators.h"
#include <tuple>

bool HD operator==(const Timehandler& lhs, const TimeDay& rhs) {
    unsigned minutes = lhs.getMinutes();
    return ((minutes / 60) == rhs.hours) && ((minutes % 60) == rhs.minutes);
}

bool HD operator==(const TimeDay& lhs, const Timehandler& rhs) { return rhs == lhs; }

bool HD operator!=(const Timehandler& lhs, const TimeDay& rhs) { return !(lhs == rhs); }

bool HD operator!=(const TimeDay& lhs, const Timehandler& rhs) { return !(lhs == rhs); }

bool HD operator<(const Timehandler& lhs, const TimeDay& rhs) {
    unsigned minutes = lhs.getMinutes();
    const char hours = static_cast<char>(minutes / 60);
    const char mins = static_cast<char>(minutes % 60);
    return std::tie(hours, mins) < std::tie(rhs.hours, rhs.minutes);
}

bool HD operator<=(const Timehandler& lhs, const TimeDay& rhs) { return !(lhs > rhs); }

bool HD operator<(const TimeDay& lhs, const Timehandler& rhs) {
    unsigned minutes = rhs.getMinutes();
    const char hours = static_cast<char>(minutes / 60);
    const char mins = static_cast<char>(minutes % 60);
    return std::tie(lhs.hours, lhs.minutes) < std::tie(hours, mins);
}

bool HD operator<=(const TimeDay& lhs, const Timehandler& rhs) { return !(lhs > rhs); }

bool HD operator>(const Timehandler& lhs, const TimeDay& rhs) {
    unsigned minutes = lhs.getMinutes();
    const char hours = static_cast<char>(minutes / 60);
    const char mins = static_cast<char>(minutes % 60);
    return std::tie(hours, mins) > std::tie(rhs.hours, rhs.minutes);
}

bool HD operator>=(const Timehandler& lhs, const TimeDay& rhs) { return !(lhs < rhs); }

bool HD operator>(const TimeDay& lhs, const Timehandler& rhs) {
    unsigned minutes = rhs.getMinutes();
    const char hours = static_cast<char>(minutes / 60);
    const char mins = static_cast<char>(minutes % 60);
    return std::tie(lhs.hours, lhs.minutes) > std::tie(hours, mins);
}

bool HD operator>=(const TimeDay& lhs, const Timehandler& rhs) { return !(lhs < rhs); }

TimeDayDuration HD operator-(const TimeDay& lhs, const Timehandler& rhs) {
    auto minslhs = lhs.getMinutes();
    auto minsrhs = rhs.getMinutes();
    return TimeDayDuration(minslhs - minsrhs);
}
