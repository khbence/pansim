#pragma once
#include <string>
#include "closurePolicy.hpp"
#include "infectionPolicy.hpp"
#include "movementPolicy.hpp"
#include "testingPolicy.hpp"
#include <memory>
#include "timeHandler.hpp"
#include "agentHandler.hpp"

class Simulation {
    std::shared_ptr<policies::ClosurePolicy> closure;
    std::shared_ptr<policies::InfectionPolicy> infection;
    std::shared_ptr<policies::MovementPolicy> movement;
    std::shared_ptr<policies::TestingPolicy> testing;
    AgentHandler agentHandler;

public:
    Simulation(std::shared_ptr<policies::ClosurePolicy> closure,
        std::shared_ptr<policies::InfectionPolicy> infection,
        std::shared_ptr<policies::MovementPolicy> movement,
        std::shared_ptr<policies::TestingPolicy> testing);
    
    void run(Timehandler start, Timehandler end, unsigned timeStep);
};
