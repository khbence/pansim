#include "runtime/simulationRuntime.h"
#include "timing.h"

#include <iostream>

int main(int argc, char** argv) {
    BEGIN_PROFILING("init");
    try {
        BEGIN_PROFILING("bootstrap");
        runtime::BootstrapResult bootstrap = runtime::bootstrapSimulation(argc, argv);
        END_PROFILING("bootstrap");
        if (bootstrap.shouldExit) {
            std::cout << bootstrap.output << std::endl;
            return bootstrap.exitCode;
        }

        END_PROFILING("init");
        BEGIN_PROFILING("runSimulation");
        bootstrap.engine->runSimulation();
        END_PROFILING("runSimulation");
        Timing::report();
    } catch (const std::exception& e) {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
