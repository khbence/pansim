#pragma once
#include "policy.hpp"
#include <memory>
#include "timeHandler.hpp"

namespace policies {
class TestingPolicy : public Policy {
public:
    TestingPolicy(ParallelStrategy parallelStrategy_p);

    virtual void performTests(Timehandler simTime, unsigned timeStep);
};

class TestingPolicyFactory : public PolicyFactory {
    template<typename... Args>
    std::unique_ptr<TestingPolicy> createPolicy(ParallelStrategy parallelStrategy_p, Args... args) {
        return std::make_unique<TestingPolicy>(parallelStrategy_p, args...);
    }
};
}// namespace policies