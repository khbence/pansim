#pragma once
#include "timeHandler.h"
#include "datatypes.h"

class Timehandler;
class TimeDayDuration;

class TimeDay {
protected:
    unsigned char hours;
    unsigned char minutes;

public:
    friend bool HD operator==(const Timehandler&, const TimeDay&);
    friend bool HD operator==(const TimeDay&, const Timehandler&);
    friend bool HD operator!=(const Timehandler&, const TimeDay&);
    friend bool HD operator!=(const TimeDay&, const Timehandler&);
    friend bool HD operator<(const Timehandler&, const TimeDay&);
    friend bool HD operator<=(const Timehandler&, const TimeDay&);
    friend bool HD operator<(const TimeDay&, const Timehandler&);
    friend bool HD operator<=(const TimeDay&, const Timehandler&);
    friend bool HD operator>(const Timehandler&, const TimeDay&);
    friend bool HD operator>=(const Timehandler&, const TimeDay&);
    friend bool HD operator>(const TimeDay&, const Timehandler&);
    friend bool HD operator>=(const TimeDay&, const Timehandler&);
    friend TimeDayDuration HD operator-(const TimeDay&, const Timehandler&);

    TimeDay HD operator+(const TimeDayDuration& dur) const;
    TimeDay& HD operator+=(const TimeDayDuration& dur);

    TimeDay HD operator-(const TimeDayDuration& dur) const;
    TimeDay& HD operator-=(const TimeDayDuration& dur);
    TimeDayDuration HD operator-(const TimeDay& other);

    bool HD operator<(const TimeDay& other) const;
    bool HD operator<=(const TimeDay& other) const;
    bool HD operator>(const TimeDay& other) const;
    bool HD operator>=(const TimeDay& other) const;
    bool HD operator==(const TimeDay& other) const;
    bool HD operator!=(const TimeDay& other) const;


    explicit HD TimeDay(double raw);
    explicit HD TimeDay(unsigned mins);
    explicit HD TimeDay(const Timehandler& t);
    unsigned HD getMinutes() const;
    unsigned HD getHours() const;

    [[nodiscard]] unsigned HD steps(unsigned timeStep) const;
    [[nodiscard]] unsigned HD getStepsUntilMidnight(unsigned timeStep) const;
    [[nodiscard]] bool HD isOverMidnight() const;
};

class TimeDayDuration : public TimeDay {
public:
    explicit HD TimeDayDuration(double raw);
    explicit HD TimeDayDuration(unsigned mins);
    [[nodiscard]] bool HD isUndefinedDuration() const;
};