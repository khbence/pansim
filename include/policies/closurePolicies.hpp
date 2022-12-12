#pragma once
#include "policy.hpp"
#include <memory>
#include "timeHandler.hpp"

namespace policies {
class ClosurePolicy : public Policy {
public:
    ClosurePolicy(ParallelStrategy parallelStrategy_p);

    virtual void midnight();
    virtual void step();
};

class ClosurePolicyFactory : public PolicyFactory {
    template<typename... Args>
    std::unique_ptr<ClosurePolicy> createPolicy(ParallelStrategy parallelStrategy_p, Args... args) {
        return std::make_unique<ClosurePolicy>(parallelStrategy_p, args...);
    }
};
}// namespace policies
