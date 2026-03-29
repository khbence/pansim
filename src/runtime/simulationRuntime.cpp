#include "runtime/simulationRuntime.h"
#include "runtime/programOptions.h"

#include "simulation.h"
#include "configTypes.h"
#include "randomGenerator.h"
#include "version.h"

#include <cxxopts.hpp>
#include <omp.h>

#include <cstdlib>
#include <random>
#include <stdexcept>
#include <utility>

namespace runtime {

namespace {

std::vector<char*> makeArgv(std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    return argv;
}

void configureRuntime(const cxxopts::ParseResult& result) {
    if (result.count("threads") != 0) {
        const int requestedThreads = result["threads"].as<int>();
        if (requestedThreads < 1) {
            throw std::runtime_error("--threads must be at least 1");
        }
        omp_set_dynamic(0);
        omp_set_num_threads(requestedThreads);
    }

    const std::uint64_t seed = result.count("seed") != 0 ? result["seed"].as<std::uint64_t>() : std::random_device{}();
    RandomGenerator::init(omp_get_max_threads(), seed);
}

} // namespace

struct SimulationEngine::Impl {
    explicit Impl(cxxopts::ParseResult&& parsedArgs)
        : args(std::move(parsedArgs)), simulation(std::make_unique<config::Simulation_t>(args)) {}

    cxxopts::ParseResult args;
    std::unique_ptr<config::Simulation_t> simulation;
};

SimulationEngine::SimulationEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

SimulationEngine::SimulationEngine(SimulationEngine&&) noexcept = default;

SimulationEngine& SimulationEngine::operator=(SimulationEngine&&) noexcept = default;

SimulationEngine::~SimulationEngine() = default;

void SimulationEngine::runSimulation() { impl_->simulation->runSimulation(); }

std::vector<unsigned> SimulationEngine::runForDay(const std::vector<std::string>& args) {
    std::vector<std::string> mutableArgs = args;
    std::vector<char*> argv = makeArgv(mutableArgs);
    return impl_->simulation->runForDay(static_cast<int>(argv.size()), argv.data());
}

void SimulationEngine::finalize() { impl_->simulation->finalize(); }

BootstrapResult bootstrapSimulation(int argc, char** argv) {
    cxxopts::Options options = defineProgramParameters();
    config::Simulation_t::addProgramParameters(options);
    cxxopts::ParseResult result = options.parse(argc, argv);

    BootstrapResult bootstrap;
    if (result.count("help") != 0) {
        bootstrap.shouldExit = true;
        bootstrap.exitCode = EXIT_SUCCESS;
        bootstrap.output = options.help();
        return bootstrap;
    }

    if (result.count("version") != 0) {
        bootstrap.shouldExit = true;
        bootstrap.exitCode = EXIT_SUCCESS;
        bootstrap.output = config::GIT_VERSION;
        return bootstrap;
    }

    configureRuntime(result);
    bootstrap.engine = std::make_unique<SimulationEngine>(std::make_unique<SimulationEngine::Impl>(std::move(result)));
    return bootstrap;
}

BootstrapResult bootstrapSimulation(const std::vector<std::string>& args) {
    std::vector<std::string> mutableArgs = args;
    std::vector<char*> argv = makeArgv(mutableArgs);
    return bootstrapSimulation(static_cast<int>(argv.size()), argv.data());
}

} // namespace runtime

#include "locationListRuntime.cpp"
#include "simulationOrchestration.cpp"
