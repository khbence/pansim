#include "infectionPolicy.hpp"
#include <logger.hpp>

policies::InfectionPolicy::InfectionPolicy(ParallelStrategy parallelStrategy_p)
    : Policy(parallelStrategy_p) {}

double policies::InfectionPolicy::getSeasonalMultiplier([[maybe_unused]] const Timehandler& t) {
    Logger::debug("InfectionPolicy::getSeasonalMultiplier");
    return 0.0;
}

void policies::InfectionPolicy::infectionsAtLocations([[maybe_unused]] Timehandler& simTime,
    [[maybe_unused]] unsigned timeStep) {
    Logger::debug("InfectionPolicy::infectionsAtLocations");
}
