#pragma once
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

template<typename T>
class DevMemory {
    thrust::device_ptr<T> p;

public:
    DevMemory(std::size_t n) : p(thrust::device_malloc<int>(N);) {}

    ~DevMemory() {
        if (p) { thrust::device_free(p); }
    }

    operator thrust::device_ptr<T>() const { return p; }
};