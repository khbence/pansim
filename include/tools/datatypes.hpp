#pragma once
#include "thrust/detail/config/device_system.h"

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#define HD __host__ __device__
#else
#define HD
#endif

namespace globalConstants {
    constexpr std::size_t MAX_STRAINS = 7;
}

// #define AGENT = 0;
// #define FROM = 1;
// #define TO = 2;