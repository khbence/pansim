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

cxxopts::Options defineProgramParameters() {
    cxxopts::Options options("covid", "An agent-based epidemic simulator");
    options.add_options()("w,weeks",
        "Length of simulation in weeks",
        cxxopts::value<unsigned>()->default_value("12"))("t,deltat",
        "Length of timestep in minutes",
        cxxopts::value<unsigned>()->default_value("10"))(
        "n,numagents", "Number of agents", cxxopts::value<int>()->default_value("-1"))("N,numlocs",
        "Number of dummy locations",
        cxxopts::value<int>()->default_value("-1"))("P,progression",
        "Path to the config file for the progression matrices.",
        cxxopts::value<std::string>()->default_value(".." + separator() + "inputFiles" + separator()
                                                     + "progressions" + separator()
                                                     + "transition_config.json"))("a,agents",
        "Agents file, for all human being in the experiment.",
        cxxopts::value<std::string>()->default_value(
            ".." + separator() + "inputFiles" + separator() + "agents.json"))("A,agentTypes",
        "List and schedule of all type fo agents.",
        cxxopts::value<std::string>()->default_value(
            ".." + separator() + "inputFiles" + separator() + "agentTypes.json"))("l,locations",
        "List of all locations in the simulation.",
        cxxopts::value<std::string>()->default_value(
            ".." + separator() + "inputFiles" + separator() + "locations.json"))("L,locationTypes",
        "List of all type of locations",
        cxxopts::value<std::string>()->default_value(
            ".." + separator() + "inputFiles" + separator() + "locationTypes.json"))("p,parameters",
        "List of all general parameters for the simulation except the "
        "progression data.",
        cxxopts::value<std::string>()->default_value(
            ".." + separator() + "inputFiles" + separator() + "parameters.json"))("c,configRandom",
        "Config file for random initialization.",
        cxxopts::value<std::string>()->default_value(".." + separator() + "inputFiles" + separator()
                                                     + "configRandom.json"))("closures",
        "List of closure rules.",
        cxxopts::value<std::string>()->default_value(
            ".." + separator() + "inputFiles" + separator() + "closureRules.json"))("r,randomStates",
        "Change the states from the agents file with the configRandom file's "
        "stateDistribution.")("outAgentStat",
        "name of the agent stat output file, if not set there will be no print",
        cxxopts::value<std::string>()->default_value(""))("diags",
        "level of diagnositcs to print",
        cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(0))));
    ;

    return options;
}

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

    RandomGenerator::init(omp_get_max_threads());
    config::Simulation_t s{ result };
    END_PROFILING("init");
    BEGIN_PROFILING("runSimulation");
    s.runSimulation();
    END_PROFILING("runSimulation");
    Timing::report();
    return EXIT_SUCCESS;
}
