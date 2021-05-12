#pragma once
#include "timeDay.h"
#include "timeHandler.h"

bool HD operator==(const Timehandler&, const TimeDay&);
bool HD operator==(const TimeDay&, const Timehandler&);
bool HD operator!=(const Timehandler&, const TimeDay&);
bool HD operator!=(const TimeDay&, const Timehandler&);
bool HD operator<(const Timehandler&, const TimeDay&);
bool HD operator<=(const Timehandler&, const TimeDay&);
bool HD operator<(const TimeDay&, const Timehandler&);
bool HD operator<=(const TimeDay&, const Timehandler&);
bool HD operator>(const Timehandler&, const TimeDay&);
bool HD operator>=(const Timehandler&, const TimeDay&);
bool HD operator>(const TimeDay&, const Timehandler&);
bool HD operator>=(const TimeDay&, const Timehandler&);
TimeDayDuration HD operator-(const TimeDay&, const Timehandler&);