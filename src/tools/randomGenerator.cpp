#include "randomGenerator.h"

std::vector<std::mt19937_64> RandomGenerator::generators;
std::uint64_t RandomGenerator::baseSeed = 0;

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cuda.h>
#include <curand_kernel.h>
__device__ curandState* dstates;
unsigned dstates_size = 0;
__global__ void setup_kernel(unsigned total, std::uint64_t seed, curandState* dstates2) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    if (id < total) curand_init(seed, id, 0, &dstates2[id]);
}
#endif

std::uint64_t RandomGenerator::deriveSeed(unsigned streamIdx) {
    std::uint64_t z = baseSeed + 0x9e3779b97f4a7c15ULL * (static_cast<std::uint64_t>(streamIdx) + 1ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

void RandomGenerator::init(unsigned agents, std::uint64_t seed) {
    baseSeed = seed;
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    curandState* devStates;
    cudaMalloc((void**)&devStates, agents * sizeof(curandState));
    cudaMemcpyToSymbol(dstates, &devStates, sizeof(curandState*));
    dstates_size = agents;
    setup_kernel<<<(agents - 1) / 128 + 1, 128>>>(agents, baseSeed, devStates);
    cudaDeviceSynchronize();
#endif
    unsigned threads = omp_get_max_threads();
    generators.clear();
    generators.reserve(threads);
    for (unsigned i = 0; i < threads; ++i) { generators.emplace_back(deriveSeed(i)); }
}

void RandomGenerator::resize(unsigned agents) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    if (dstates_size < agents) {
        curandState* devStates;
        cudaMalloc((void**)&devStates, agents * sizeof(curandState));
        curandState* devStates_old;
        cudaMemcpyFromSymbol(&devStates_old, dstates, sizeof(curandState*));
        cudaMemcpy(devStates, devStates_old, dstates_size * sizeof(curandState), cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(dstates, &devStates, sizeof(curandState*));
        setup_kernel<<<(agents - dstates_size - 1) / 128 + 1, 128>>>(agents - dstates_size, baseSeed, devStates + dstates_size);
        dstates_size = agents;
        cudaDeviceSynchronize();
    }
#endif
    if (generators.size() < omp_get_max_threads()) {
        unsigned threads = omp_get_max_threads();
        generators.reserve(threads);
        for (unsigned i = generators.size(); i < threads; ++i) { generators.emplace_back(deriveSeed(i)); }
    }
}
