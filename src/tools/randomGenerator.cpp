#include "hip/hip_runtime.h"
#include "randomGenerator.h"

std::vector<std::mt19937_64> RandomGenerator::generators;

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
__device__ hiprandState* dstates;
unsigned dstates_size = 0;
__global__ void setup_kernel(unsigned total, hiprandState* dstates2) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    if (id < total) hiprand_init(1234, id, 0, &dstates2[id]);
}
#endif

void RandomGenerator::init(unsigned agents) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
    hiprandState* devStates;
    hipMalloc((void**)&devStates, agents * sizeof(hiprandState));
    hipMemcpyToSymbol(HIP_SYMBOL(dstates), &devStates, sizeof(hiprandState*));
    dstates_size = agents;
    setup_kernel<<<(agents - 1) / 128 + 1, 128>>>(agents, devStates);
    hipDeviceSynchronize();
#endif
    unsigned threads = omp_get_max_threads();
    generators.reserve(threads);
    std::random_device rd;
    for (unsigned i = 0; i < threads; ++i) { generators.emplace_back(rd()); }
}

void RandomGenerator::resize(unsigned agents) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
    if (dstates_size < agents) {
        hiprandState* devStates;
        hipMalloc((void**)&devStates, agents * sizeof(hiprandState));
        hiprandState* devStates_old;
        hipMemcpyFromSymbol(&devStates_old, HIP_SYMBOL(dstates), sizeof(hiprandState*));
        hipMemcpy(devStates, devStates_old, dstates_size * sizeof(hiprandState), hipMemcpyDeviceToDevice);
        hipMemcpyToSymbol(HIP_SYMBOL(dstates), &devStates, sizeof(hiprandState*));
        setup_kernel<<<(agents - dstates_size - 1) / 128 + 1, 128>>>(agents - dstates_size, devStates + dstates_size);
        dstates_size = agents;
        hipDeviceSynchronize();
    }
#endif
    if (generators.size() < omp_get_max_threads()) {
        unsigned threads = omp_get_max_threads();
        generators.reserve(threads);
        std::random_device rd;
        for (unsigned i = generators.size(); i < threads; ++i) { generators.emplace_back(rd()); }
    }
}