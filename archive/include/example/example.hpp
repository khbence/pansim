#pragma once
#include <vector>
#include <thrust/device_vector.h>

class Example {
    thrust::device_vector<double> data;

public:
    explicit Example(const std::vector<double>& dataP);
    [[nodiscard]] double getSum() const;
};