#include "movementPolicy.hpp"
#include <logger.hpp>

policies::MovementPolicy::MovementPolicy(ParallelStrategy parallelStrategy_p)
    : Policy(parallelStrategy_p) {}

void policies::MovementPolicy::planLocations([[maybe_unused]] Timehandler simTime,
    [[maybe_unused]] unsigned timeStep) {
    Logger::debug("MovementPolicy::planLocations");
}

void policies::MovementPolicy::movement([[maybe_unused]] Timehandler simTime,
    [[maybe_unused]] unsigned timeStep) {
    Logger::debug("MovementPolicy::movement");
}
