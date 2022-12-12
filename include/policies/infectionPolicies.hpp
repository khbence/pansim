#pragma once
#include "policy.hpp"
#include <memory>
#include "timeHandler.hpp"

namespace policies {
class InfectionPolicy : public Policy {
public:
    InfectionPolicy(ParallelStrategy parallelStrategy_p);

    virtual double getSeasonalMultiplier(const Timehandler& t);
    virtual void infectionsAtLocations(Timehandler& simTime, unsigned timeStep);
};

class InfectionPolicyFactory : public PolicyFactory {
    template<typename... Args>
    std::unique_ptr<InfectionPolicy> createPolicy(ParallelStrategy parallelStrategy_p, Args... args) {
        return std::make_unique<InfectionPolicy>(parallelStrategy_p, args...);
    }
};
}// namespace policies