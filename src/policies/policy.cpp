#include "policy.hpp"

namespace policies {
Policy::Policy(ParallelStrategy parallelStrategy_p) : parallelStrategy(parallelStrategy_p) {}

ParallelStrategy Policy::getParallelStrategy() const { return parallelStrategy; }
}