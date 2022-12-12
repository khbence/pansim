#include "testingPolicy.hpp"
#include <logger.hpp>

policies::TestingPolicy::TestingPolicy(ParallelStrategy parallelStrategy_p) : Policy(parallelStrategy_p) {}

void policies::TestingPolicy::performTests([[maybe_unused]] Timehandler simTime, [[maybe_unused]] unsigned timeStep) {
    Logger::debug("TestingPolicy::performTests");
}
