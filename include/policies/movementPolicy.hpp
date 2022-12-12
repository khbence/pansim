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
    virtual ~MovementPolicy() = default;
};

class MovementPolicyFactory : public PolicyFactory {
public:
    template<typename... Args>
    static std::shared_ptr<MovementPolicy> createPolicy(ParallelStrategy parallelStrategy_p, Args... args) {
        return std::make_shared<MovementPolicy>(parallelStrategy_p, args...);
    }
};
}// namespace policies