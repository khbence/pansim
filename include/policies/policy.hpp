#pragma once

namespace policies {
enum class ParallelStrategy { GENERAL, SINGLE, CPU_PARALLEL, GPU_PARALLEL };

class Policy {
    ParallelStrategy parallelStrategy = ParallelStrategy::GENERAL;

protected:
    Policy() = default;
    Policy(ParallelStrategy parallelStrategy);

public:
    ParallelStrategy getParallelStrategy() const;
    virtual ~Policy() = default;
};

class PolicyFactory {};
}// namespace policies