#include "simulation.h"
#include "configTypes.h"
#include "movementPolicies.h"
#include "infectionPolicies.h"
#include <iostream>
#include "agentMeta.h"
// for testing
#include <inputJSON.h>
#include <random>
#include "randomGenerator.h"
#include <omp.h>
#include "timing.h"
#include <cxxopts.hpp>
#include "smallTools.h"
#include "datatypes.h"
#include "version.h"

int main(int argc, char** argv) {
    BEGIN_PROFILING("init");

    auto options = defineProgramParameters();
    config::Simulation_t::addProgramParameters(options);

    options.add_options()("h,help", "Print usage");
    options.add_options()("version", "Print version");
    cxxopts::ParseResult result = options.parse(argc, argv);
    if (result.count("help") != 0) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    } else if (result.count("version") != 0) {
        std::cout << config::GIT_VERSION << std::endl;
        return EXIT_SUCCESS;
    }

    BEGIN_PROFILING("Device/RNG init");
    RandomGenerator::init(omp_get_max_threads());
    END_PROFILING("Device/RNG init");
    try {
        config::Simulation_t s{ result };
        END_PROFILING("init");
        BEGIN_PROFILING("runSimulation");
        s.runSimulation();
        END_PROFILING("runSimulation");
        Timing::report();
    } catch (const init::ProgramInit& e) {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
