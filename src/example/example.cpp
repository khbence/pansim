#include "example.hpp"
#include <thrust/host_vector.h>

Example::Example(const std::vector<double>& dataP) {
    thrust::host_vector<double> tmp{dataP};
    data = tmp;
}

double Example::getSum() const {
    return data[0];
}