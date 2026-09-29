#pragma once
#ifndef THRUST_DEVICE_SYSTEM
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
#endif

#if !defined(__CUDACC__) && !defined(__CUDA_ARCH__)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define HD __host__ __device__
#else
#define HD
#endif

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#define MAX_STRAINS 7
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) < (b) ? (b) : (a))
