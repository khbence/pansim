#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace runtime {

class SimulationEngine {
public:
    struct Impl;

    explicit SimulationEngine(std::unique_ptr<Impl> impl);
    SimulationEngine(SimulationEngine&&) noexcept;
    SimulationEngine& operator=(SimulationEngine&&) noexcept;
    ~SimulationEngine();

    void runSimulation();
    std::vector<unsigned> runForDay(const std::vector<std::string>& args);
    void finalize();

private:
    std::unique_ptr<Impl> impl_;
};

struct BootstrapResult {
    bool shouldExit = false;
    int exitCode = 0;
    std::string output;
    std::unique_ptr<SimulationEngine> engine;
};

BootstrapResult bootstrapSimulation(int argc, char** argv);
BootstrapResult bootstrapSimulation(const std::vector<std::string>& args);

} // namespace runtime
