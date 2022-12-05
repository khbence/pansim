#pragma once
#include "cxxopts.hpp"

namespace policies {
class AbstractPolicy {
public:
    static void addProgramParameters(cxxopts::Options& options);
};
}