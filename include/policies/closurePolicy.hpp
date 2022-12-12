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
    virtual ~ClosurePolicy() = default;
};

class ClosurePolicyFactory : public PolicyFactory {
public:
    template<typename... Args>
    static std::shared_ptr<ClosurePolicy> createPolicy(ParallelStrategy parallelStrategy_p, Args... args) {
        return std::make_shared<ClosurePolicy>(parallelStrategy_p, args...);
    }
};
}// namespace policies
