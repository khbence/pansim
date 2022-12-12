#pragma once
#include "policy.hpp"
#include <memory>
#include "timeHandler.hpp"

namespace policies {
class MovementPolicy : public Policy {
public:
    MovementPolicy(ParallelStrategy parallelStrategy_p);

    virtual void planLocations(Timehandler simTime, unsigned timeStep);
    void movement(Timehandler simTime, unsigned timeStep);
};

class MovementPolicyFactory : public PolicyFactory {
    template<typename... Args>
    std::unique_ptr<MovementPolicy> createPolicy(ParallelStrategy parallelStrategy_p, Args... args) {
        return std::make_unique<MovementPolicy>(parallelStrategy_p, args...);
    }
};
}// namespace policies