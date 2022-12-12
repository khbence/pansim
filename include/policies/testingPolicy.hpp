#pragma once
#include "policy.hpp"
#include <memory>
#include "timeHandler.hpp"

namespace policies {
class TestingPolicy : public Policy {
public:
    TestingPolicy(ParallelStrategy parallelStrategy_p);

    virtual void performTests(Timehandler simTime, unsigned timeStep);
    virtual ~TestingPolicy() = default;
};

class TestingPolicyFactory : public PolicyFactory {
public:
    template<typename... Args>
    static std::shared_ptr<TestingPolicy> createPolicy(ParallelStrategy parallelStrategy_p, Args... args) {
        return std::make_shared<TestingPolicy>(parallelStrategy_p, args...);
    }
};
}// namespace policies