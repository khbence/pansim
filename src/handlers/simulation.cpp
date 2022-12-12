#include "simulation.hpp"

Simulation::Simulation(std::shared_ptr<policies::ClosurePolicy> closure_p,
    std::shared_ptr<policies::InfectionPolicy> infection_p,
    std::shared_ptr<policies::MovementPolicy> movement_p,
    std::shared_ptr<policies::TestingPolicy> testing_p)
    : closure(closure_p), infection(infection_p), movement(movement_p), testing(testing_p) {}

void Simulation::run(Timehandler start, Timehandler end, unsigned timeStep) {
    for (Timehandler simTime = start; simTime < end; simTime += timeStep) {
        if (simTime.isMidnight()) {
            if (simTime.getTimestamp() > 0) {
                testing->performTests(simTime, timeStep);
                // TODO update agents, probably in agent handler
            }
            closure->midnight();
            movement->planLocations(simTime, timeStep);
        }
        movement->movement(simTime, timeStep);
        closure->step();
    }
}
