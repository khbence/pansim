#include "runtime/programOptions.h"

#include "smallTools.h"

#include <cstdint>
#include <string>

cxxopts::Options defineProgramParameters() {
    cxxopts::Options options("covid", "An agent-based epidemic simulator");
    options.add_options()("w,weeks", "Length of simulation in weeks", cxxopts::value<unsigned>()->default_value("12"))(
        "t,deltat", "Length of timestep in minutes", cxxopts::value<unsigned>()->default_value("10"))(
        "n,numagents", "Number of agents", cxxopts::value<int>()->default_value("-1"))(
        "N,numlocs", "Number of dummy locations", cxxopts::value<int>()->default_value("-1"))("P,progression",
        "Path to the config file for the progression matrices.",
        cxxopts::value<std::string>()->default_value(
            "inputConfigFiles" + separator() + "progressions" + separator() + "transition_config.json"))("a,agents",
        "Agents file, for all human being in the experiment.",
        cxxopts::value<std::string>()->default_value("inputRealExample" + separator() + "agents.json"))("A,agentTypes",
        "List and schedule of all type fo agents.",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "agentTypes.json"))("l,locations",
        "List of all locations in the simulation.",
        cxxopts::value<std::string>()->default_value("inputRealExample" + separator() + "locations.json"))("L,locationTypes",
        "List of all type of locations",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "locationTypes.json"))("p,parameters",
        "List of all general parameters for the simulation except the "
        "progression data.",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "parameters.json"))("c,configRandom",
        "Config file for random initialization.",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "configRandom.json"))("closures",
        "List of closure rules.",
        cxxopts::value<std::string>()->default_value("inputConfigFiles" + separator() + "closureRules.json"))("r,randomStates",
        "Change the states from the agents file with the configRandom file's "
        "stateDistribution.")("outAgentStat",
        "name of the agent stat output file, if not set there will be no print",
        cxxopts::value<std::string>()->default_value(""))(
        "seed",
        "Base RNG seed used for deterministic random streams.",
        cxxopts::value<std::uint64_t>())(
        "threads",
        "Override the OpenMP thread count before simulation setup.",
        cxxopts::value<int>())(
        "diags", "level of diagnositcs to print", cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(0))));

    options.add_options()("h,help", "Print usage");
    options.add_options()("version", "Print version");

    return options;
}
