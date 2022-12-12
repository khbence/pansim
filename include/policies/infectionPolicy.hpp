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
    virtual ~InfectionPolicy() = default;
};

class InfectionPolicyFactory : public PolicyFactory {
public:
    template<typename... Args>
    static std::shared_ptr<InfectionPolicy> createPolicy(ParallelStrategy parallelStrategy_p, Args... args) {
        return std::make_shared<InfectionPolicy>(parallelStrategy_p, args...);
    }
};
}// namespace policies