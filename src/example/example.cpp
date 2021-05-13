#include "example.hpp"
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

Example::Example(const std::vector<double>& dataP) {
    thrust::host_vector<double> tmp{ dataP };
    data = tmp;
}

double Example::getSum() const { return thrust::reduce(data.begin(), data.end(), 0, thrust::plus<int>()); }