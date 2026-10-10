#pragma once
#include <random>
#include <cstdint>
#include <cmath>
#include <vector>
#include <omp.h>
#include "datatypes.h"
#include "timing.h"
#include <limits.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cuda.h>
#include <curand_kernel.h>
extern __device__ curandState* dstates;
#endif
class RandomGenerator {
    static std::vector<std::mt19937_64> generators;
    static std::uint64_t baseSeed;

    [[nodiscard]] static std::uint64_t deriveSeed(unsigned streamIdx);

public:
    static void init(unsigned agents, std::uint64_t seed);
    static void resize(unsigned agents);

    [[nodiscard]] static __host__ thrust::host_vector<float> fillUnitf(unsigned size) {
        Timing::startTimer("RandomGenerator::fillUnitf");
        thrust::host_vector<float> tmp(size);
        std::uniform_real_distribution<double> dis(0, 1);
        //#pragma omp parallel for private(dis)
        for (int i = 0; i < size; i++) { tmp[i] = dis(generators[0]); }
        Timing::stopTimer("RandomGenerator::fillUnitf");
        return tmp;
    }

    [[nodiscard]] static __host__ __device__ double randomUnit() {
#ifdef __CUDA_ARCH__
        return curand_uniform_double(&dstates[threadIdx.x + blockIdx.x * blockDim.x]);
#else
        std::uniform_real_distribution<double> dis(0, 1);
        return dis(generators[omp_get_thread_num()]);
#endif
    }
#if 0
    [[nodiscard]] static __host__ double randomUnit() {
        std::uniform_real_distribution<double> dis(0, 1);
        return dis(generators[omp_get_thread_num()]);    
    }
#endif

    [[nodiscard]] static __host__ __device__ double randomReal(double max) {
#ifdef __CUDA_ARCH__
        return max * curand_uniform_double(&dstates[threadIdx.x + blockIdx.x * blockDim.x]);
#else
        std::uniform_real_distribution<double> dis(0, max);
        return dis(generators[omp_get_thread_num()]);
#endif
    }
#if 0
    [[nodiscard]] static __host__ double randomReal(double max) {
        std::uniform_real_distribution<double> dis(0, max);
        return dis(generators[omp_get_thread_num()]);
    }
#endif

    [[nodiscard]] static __host__ __device__ unsigned randomUnsigned(unsigned max) {
        if (max == 0) return 0u;
        --max;
#ifdef __CUDA_ARCH__
        return curand(&dstates[threadIdx.x + blockIdx.x * blockDim.x]) % (max + 1);
#else
        std::uniform_int_distribution<unsigned> dis(0, max);
        return dis(generators[omp_get_thread_num()]);
#endif
    }
#if 0
    [[nodiscard]] static __host__ unsigned randomUnsigned(unsigned max) {
        --max;
        std::uniform_int_distribution<unsigned> dis(0, max);
        return dis(generators[omp_get_thread_num()]);
    }
#endif

    // Gamma(shape, scale=1/shape), so the mean is 1. shape <= 0 or very large returns 1.
    // Marsaglia-Tsang for shape >= 1, with the Ahrens boost when shape < 1.
    [[nodiscard]] static __host__ __device__ double gammaMeanOne(double shape) {
        if (!(shape > 0.0) || shape >= 1.0e6) return 1.0;
        double extra = 1.0;
        double alpha = shape;
        if (alpha < 1.0) {
            double uBoost = randomUnit();
            if (uBoost < 1e-12) uBoost = 1e-12;
            extra = pow(uBoost, 1.0 / alpha);
            alpha += 1.0;
        }
        const double d = alpha - 1.0 / 3.0;
        const double c = 1.0 / sqrt(9.0 * d);
        double g = 0.0;
        while (true) {
            double x;
            double v;
            do {
                double u1 = randomUnit();
                if (u1 < 1e-12) u1 = 1e-12;
                double u2 = randomUnit();
                x = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            double u = randomUnit();
            double x2 = x * x;
            if (u < 1.0 - 0.0331 * x2 * x2 || log(u) < 0.5 * x2 + d * (1.0 - v + log(v))) {
                g = d * v * extra;
                break;
            }
        }
        const double cap = 1.0e4 * shape;
        if (!(g > 0.0) || g > cap) g = cap;
        return g / shape;
    }

    [[nodiscard]] static __host__ __device__ int geometric(double p) {
#ifdef __CUDA_ARCH__
        double _M_p = p;
        double _M_log_1_p = log(1.0 - _M_p);
        const double __naf = (1 - __DBL_EPSILON__) / 2;
        const double __thr = __INT_MAX__ + __naf;
        double __cand;
        do
            __cand = floor(log(1.0 - curand_uniform_double(&dstates[threadIdx.x + blockIdx.x * blockDim.x])) / _M_log_1_p);
        while (__cand >= __thr);
        return int(__cand + __naf);
#else
        std::geometric_distribution<> dis(p);
        return dis(generators[omp_get_thread_num()]);
#endif
    }
#if 0
    [[nodiscard]] static __host__ int geometric(double p) {
        std::geometric_distribution<> dis(p);
        return dis(generators[omp_get_thread_num()]);
    }
#endif
};
