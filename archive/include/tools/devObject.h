#pragma once
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

namespace detail {
    namespace DevObject {
        template<typename T, typename... ARGS>
        __global__ void AllocateObject(thrust::device_ptr<T> p, ARGS... args) {
            new (p.get()) T(args...);
        }

        template<typename T>
        __global__ void DeleteObject(thrust::device_ptr<T> p) {
            p->~T();
        }
    }// namespace DevObject
}// namespace detail

template<typename T>
class DevObject {
    thrust::device_ptr<T> p;

    template<typename... ARGS>
    DevObject(std::size_t n, ARGS... args) : p(thrust::device_malloc(n)) {
        detail::DevObject::AllocateObject<T><<<1, 1>>>(p, args...);
        cudaDeviceSyncronize();
    }

    ~DevObject() {
        if (p) {
            detail::DevObject::DeleteObject<T><<<1, 1>>>(p);
            cudaDeviceSyncronize();
        }
    }

    T* get() const { return p.get(); }
};