#pragma once
#ifndef THRUST_DEVICE_SYSTEM
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
#endif

#define HD __host__ __device__

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#define MAX_STRAINS 4
#define MIN(a, b) (a) < (b) ? (a) : (b)

#define AGENT = 0;
#define FROM = 1;
#define TO = 2;