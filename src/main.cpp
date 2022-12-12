#include <iostream>
#include "simulation.hpp"
#include "closurePolicy.hpp"
#include "infectionPolicy.hpp"
#include "movementPolicy.hpp"
#include "testingPolicy.hpp"
#include <memory>
#include "timeHandler.hpp"
#include "logger.hpp"

int main([[maybe_unused]] int argc, [[maybe_unused]] const char* argv[]) {
    Logger::setLowestLevel(LogLevel::INFO);
    policies::ParallelStrategy parallelStrategy = policies::ParallelStrategy::GENERAL;
    auto closure = policies::ClosurePolicyFactory::createPolicy(parallelStrategy);
    auto infection = policies::InfectionPolicyFactory::createPolicy(parallelStrategy);
    auto movement = policies::MovementPolicyFactory::createPolicy(parallelStrategy);
    auto testing = policies::TestingPolicyFactory::createPolicy(parallelStrategy);
    Simulation sim(closure, infection, movement, testing);
    Timehandler startTime{10, 0, Days::MONDAY, 1};
    Timehandler endTime{10, 1, Days::MONDAY, 1};
    sim.run(startTime, endTime, 10);
    return EXIT_SUCCESS;
}
