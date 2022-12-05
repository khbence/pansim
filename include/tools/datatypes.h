#pragma once
#include "thrust/detail/config/device_system.h"

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#define HD __host__ __device__
#else
#define HD
#endif

// #define AGENT = 0;
// #define FROM = 1;
// #define TO = 2;