#include "closurePolicy.hpp"
#include <logger.hpp>

policies::ClosurePolicy::ClosurePolicy(ParallelStrategy parallelStrategy_p) : Policy(parallelStrategy_p) {}

void policies::ClosurePolicy::midnight() {
    Logger::debug("ClosurePolicy::midnight");
}

void policies::ClosurePolicy::step() {
    Logger::debug("ClosurePolicy::step");
}
