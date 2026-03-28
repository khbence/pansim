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

    cxxopts::ParseResult result = options.parse(argc, argv);
    if (result.count("help") != 0) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    } else if (result.count("version") != 0) {
        std::cout << config::GIT_VERSION << std::endl;
        return EXIT_SUCCESS;
    }

    if (result.count("threads") != 0) {
        const int requestedThreads = result["threads"].as<int>();
        if (requestedThreads < 1) {
            std::cerr << "--threads must be at least 1\n";
            return EXIT_FAILURE;
        }
        omp_set_dynamic(0);
        omp_set_num_threads(requestedThreads);
    }

    const std::uint64_t seed = result.count("seed") != 0 ? result["seed"].as<std::uint64_t>() : std::random_device{}();

    BEGIN_PROFILING("Device/RNG init");
    RandomGenerator::init(omp_get_max_threads(), seed);
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
