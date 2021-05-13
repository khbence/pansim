#include "timeHandler.h"


// + operators
Timehandler Timehandler::operator+(unsigned steps) const {
    Timehandler ret = *this;
    ret += steps;
    return ret;
}

Timehandler& Timehandler::operator+=(unsigned steps) {
    counter += steps;
    current += steps * timeStep;
    return *this;
}

Timehandler Timehandler::operator+(const TimeDayDuration& dur) const {
    auto mins = dur.getMinutes();
    return this->operator+(static_cast<unsigned>(mins / timeStep.count()));
}

Timehandler& Timehandler::operator+=(const TimeDayDuration& dur) {
    auto mins = dur.getMinutes();
    return this->operator+=(static_cast<unsigned>(mins / timeStep.count()));
}

// - operators
Timehandler Timehandler::operator-(unsigned steps) const {
    Timehandler ret = *this;
    ret -= steps;
    return ret;
}

Timehandler& Timehandler::operator-=(unsigned steps) {
    // doesn't handle if steps > counter, it will not happen hopefully for
    // various reasons
    counter -= steps;
    current -= steps * timeStep;
    return *this;
}

Timehandler Timehandler::operator-(const TimeDayDuration& dur) const {
    auto mins = dur.getMinutes();
    return this->operator-(static_cast<unsigned>(mins / timeStep.count()));
}

Timehandler& Timehandler::operator-=(const TimeDayDuration& dur) {
    auto mins = dur.getMinutes();
    return this->operator-=(static_cast<unsigned>(mins / timeStep.count()));
}


[[nodiscard]] std::vector<Days> Timehandler::parseDays(const std::string& rawDays) {
    std::string day;
    std::vector<Days> result;
    std::transform(rawDays.begin(), rawDays.end(), std::back_inserter(day), [](char c) {
        return std::toupper(c);
    });
    if (day == "ALL") {
        result = decltype(result){ Days::MONDAY,
            Days::TUESDAY,
            Days::WEDNESDAY,
            Days::THURSDAY,
            Days::FRIDAY,
            Days::SATURDAY,
            Days::SUNDAY };
    } else if (day == "WEEKDAYS") {
        result = decltype(result){
            Days::MONDAY,
            Days::TUESDAY,
            Days::WEDNESDAY,
            Days::THURSDAY,
            Days::FRIDAY,
        };
    } else if (day == "WEEKENDS") {
        result = decltype(result){ Days::SATURDAY, Days::SUNDAY };
    } else if (day == "MONDAY") {
        result.push_back(Days::MONDAY);
    } else if (day == "TUESDAY") {
        result.push_back(Days::TUESDAY);
    } else if (day == "WEDNESDAY") {
        result.push_back(Days::WEDNESDAY);
    } else if (day == "THURSDAY") {
        result.push_back(Days::THURSDAY);
    } else if (day == "FRIDAY") {
        result.push_back(Days::FRIDAY);
    } else if (day == "SATURDAY") {
        result.push_back(Days::SATURDAY);
    } else if (day == "SUNDAY") {
        result.push_back(Days::SUNDAY);
    } else {
        throw IOAgentTypes::InvalidDayInSchedule(rawDays);
    }
    return result;
}

Timehandler::Timehandler(unsigned timeStep_p, unsigned weeksInTheFuture, Days _startDay)
    : timeStep(std::chrono::minutes(timeStep_p)),
      current(nextMidnight() + std::chrono::hours(hoursPerWeek * weeksInTheFuture)),
      stepsPerDay(minsPerDay / timeStep_p), startDay(_startDay) {
    if (minsPerDay % timeStep_p != 0) { throw init::BadTimeStep(timeStep_p); }
    dayOffset = (unsigned)getDay() > (unsigned)startDay ? ((unsigned)startDay+7)-(unsigned)getDay() : (unsigned)startDay - (unsigned)getDay(); 
}

unsigned HD Timehandler::getStepsUntilMidnight() const {
    return stepsPerDay - (counter % stepsPerDay);
}

Timehandler Timehandler::getNextMidnight() const {
    Timehandler ret = *this;
    unsigned steps = ret.getStepsUntilMidnight();
    ret += steps;
    return ret;
}

unsigned HD Timehandler::getMinutes() const { return (counter % stepsPerDay) * timeStep.count(); }

unsigned HD Timehandler::getTimestamp() const { return counter; }

Days Timehandler::getDay() const {
    time_t tt = std::chrono::system_clock::to_time_t(current);
    tm* date = std::localtime(&tt);
    date->tm_wday = date->tm_wday==0 ? 6 : date->tm_wday - 1;
    //date->tm_wday = date->tm_wday==0 ? 6 : date->tm_wday - 1;
    return static_cast<Days>((date->tm_wday + dayOffset) % 7);
}