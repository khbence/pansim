#include "example.hpp"
#include <iostream>
#include <vector>

int main([[maybe_unused]] int argc, [[maybe_unused]] char const *argv[]) {
    std::vector<double> data{0.5, 0.8, 1.5, 1.8, 2.0, 1.8};
    Example e{data};
    std::cout << e.getSum() << std::endl;
    return EXIT_SUCCESS;
}
