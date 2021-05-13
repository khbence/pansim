#include "timeDay.h"
#include <cmath>
#include <limits>

TimeDay HD TimeDay::operator+(const TimeDayDuration& dur) const {
    TimeDay ret = *this;
    ret += dur;
    return ret;
}

TimeDay& HD TimeDay::operator+=(const TimeDayDuration& dur) {
    unsigned mins = dur.getMinutes();
    minutes += mins % 60;
    auto overflow = minutes / 60;
    minutes %= 60;
    hours += (mins / 60) + overflow;
    return *this;
}

TimeDay HD TimeDay::operator-(const TimeDayDuration& dur) const {
    TimeDay ret = *this;
    ret -= dur;
    return ret;
}

TimeDay& HD TimeDay::operator-=(const TimeDayDuration& dur) {
    auto minsDur = dur.getMinutes();
    auto minsThis = getMinutes();
    auto mins = minsThis - minsDur;
    minutes = mins % 60;
    hours = mins / 60;
    return *this;
}

TimeDayDuration HD TimeDay::operator-(const TimeDay& other) {
    auto minsThis = getMinutes();
    auto minsOther = other.getMinutes();
    auto mins = minsThis - minsOther;
    return TimeDayDuration(mins);
}

bool HD TimeDay::operator<(const TimeDay& other) const {
    return std::tie(hours, minutes) < std::tie(other.hours, other.minutes);
}

bool HD TimeDay::operator<=(const TimeDay& other) const { return !(*this > other); }

bool HD TimeDay::operator>(const TimeDay& other) const {
    return std::tie(hours, minutes) > std::tie(other.hours, other.minutes);
}

bool HD TimeDay::operator>=(const TimeDay& other) const { return !(*this < other); }

bool HD TimeDay::operator==(const TimeDay& other) const {
    return (hours == other.hours) && (minutes == other.minutes);
}

bool HD TimeDay::operator!=(const TimeDay& other) const { return !((*this) == other); }

HD TimeDay::TimeDay(double raw)
    : hours(static_cast<decltype(hours)>(raw)),
      minutes(
          static_cast<decltype(minutes)>(std::round(((raw - static_cast<int>(raw)) / 0.6) * 60))) {
    if (raw == -1.0) { hours = std::numeric_limits<decltype(hours)>::max(); }
}

HD TimeDay::TimeDay(unsigned mins) : hours(mins / 60), minutes(mins % 60) {}

HD TimeDay::TimeDay(const Timehandler& t) : TimeDay(t.getMinutes()) {}

unsigned HD TimeDay::getMinutes() const {
    return static_cast<unsigned>(
        (static_cast<unsigned>(hours) * 60) + static_cast<unsigned>(minutes));
}

unsigned HD TimeDay::getHours() const { return static_cast<unsigned>(hours); }

[[nodiscard]] unsigned HD TimeDay::steps(unsigned timeStep) const {
    return getMinutes() / timeStep;
};

[[nodiscard]] unsigned HD TimeDay::getStepsUntilMidnight(unsigned timeStep) const {
    return (1440 - getMinutes()) / timeStep;
};

[[nodiscard]] bool HD TimeDay::isOverMidnight() const { return hours >= 24; }

HD TimeDayDuration::TimeDayDuration(double raw) : TimeDay(raw) {}

HD TimeDayDuration::TimeDayDuration(unsigned mins) : TimeDay(mins) {}

bool HD TimeDayDuration::isUndefinedDuration() const {
    return hours == std::numeric_limits<decltype(hours)>::max();
}