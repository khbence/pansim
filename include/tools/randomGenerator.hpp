#pragma once
#include <random>
#include <vector>
#include <omp.h>
#include "datatypes.hpp"
#include <limits.h>
#include "thrust/host_vector.h"

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cuda.h>
#include <curand_kernel.h>
extern __device__ curandState* dstates;
#endif
class RandomGenerator {
    static std::vector<std::mt19937_64> generators;

public:
    static void init(unsigned agents);
    static void resize(unsigned agents);

    [[nodiscard]] static thrust::host_vector<float> fillUnitf(std::size_t size) {
        thrust::host_vector<float> tmp(size);
        std::uniform_real_distribution<float> dis(0, 1);
        //#pragma omp parallel for private(dis)
        for (std::size_t i = 0; i < size; i++) { tmp[i] = dis(generators[0]); }
        return tmp;
    }

    template<typename FloatType = double>
    [[nodiscard]] static HD FloatType randomUnit() {
#ifdef __CUDA_ARCH__
        return curand_uniform_double(&dstates[threadIdx.x + blockIdx.x * blockDim.x]);
#else
        std::uniform_real_distribution<FloatType> dis(0, 1);
        return dis(generators[static_cast<std::size_t>(omp_get_thread_num())]);
#endif
    }
#if 0
    template<typename FloatType = double>
    [[nodiscard]] static __host__ FloatType randomUnit() {
        std::uniform_real_distribution<FloatType> dis(0, 1);
        return dis(generators[static_cast<std::size_t>(omp_get_thread_num())]);    
    }
#endif

    template<typename FloatType = double>
    [[nodiscard]] static HD FloatType randomReal(FloatType max) {
#ifdef __CUDA_ARCH__
        return max * curand_uniform_double(&dstates[threadIdx.x + blockIdx.x * blockDim.x]);
#else
        std::uniform_real_distribution<FloatType> dis(0, max);
        return dis(generators[static_cast<std::size_t>(omp_get_thread_num())]);
#endif
    }
#if 0
    template<typename FloatType = double>
    [[nodiscard]] static __host__ FloatType randomReal(FloatType max) {
        std::uniform_real_distribution<FloatType> dis(0, max);
        return dis(generators[static_cast<std::size_t>(omp_get_thread_num())]);
    }
#endif
    template<typename IntegerType = unsigned>
    [[nodiscard]] static HD IntegerType randomUnsigned(IntegerType max) {
        if (max == 0) return 0;
        --max;
#ifdef __CUDA_ARCH__
        return curand(&dstates[threadIdx.x + blockIdx.x * blockDim.x]) % (max + 1);
#else
        std::uniform_int_distribution<IntegerType> dis(0, max);
        return dis(generators[static_cast<std::size_t>(omp_get_thread_num())]);
#endif
    }
#if 0
    template<typename IntegerType = unsigned>
    [[nodiscard]] static __host__ IntegerType randomUnsigned(IntegerType max) {
        --max;
        std::uniform_int_distribution<IntegerType> dis(0, max);
        return dis(generators[static_cast<std::size_t>(omp_get_thread_num())]);
    }
#endif

    template<typename FloatType = double, typename IntegerType = int>
    [[nodiscard]] static HD IntegerType geometric(FloatType p) {
#ifdef __CUDA_ARCH__
        FloatType _M_p = p;
        FloatType _M_log_1_p = log(1.0 - _M_p);
        const FloatType __naf = (1 - __DBL_EPSILON__) / 2;
        const FloatType __thr = __INT_MAX__ + __naf;
        FloatType __cand;
        do
            __cand = floor(log(1.0 - curand_uniform_double(&dstates[threadIdx.x + blockIdx.x * blockDim.x])) / _M_log_1_p);
        while (__cand >= __thr);
        return static_cast<IntegerType>(__cand + __naf);
#else
        std::geometric_distribution<> dis(p);
        return static_cast<IntegerType>(dis(generators[static_cast<std::size_t>(omp_get_thread_num())]));
#endif
    }
#if 0
    template<typename FloatType = double, typename IntegerType = int>
    [[nodiscard]] static __host__ IntegerType geometric(FloatType p) {
        std::geometric_distribution<> dis(p);
        return static_cast<IntegerType>(dis(generators[static_cast<std::size_t>(omp_get_thread_num())]));
    }
#endif
};