#pragma once
#include <array>
#include "globalStates.hpp"
#include <vector>

class BasicLengthAbstract {
protected:
    class LengthOfState {
        std::array<float, globalConstants::MAX_STRAINS> avgLength;
        std::array<float, globalConstants::MAX_STRAINS> maxLength;
        std::array<double, globalConstants::MAX_STRAINS> p;

        [[nodiscard]] static double expectedLength(double p, float max);
        [[nodiscard]] static double calculateModifiedP(double p, float avgLength, float maxLength);

    public:
        LengthOfState() = default;
        LengthOfState(std::vector<float> avgLength_p, std::vector<float> maxLength_p);
        [[nodiscard]] HD int calculateDays(int variant) const;
    };

    BasicLengthAbstract(std::size_t n);

public:
    unsigned numStates;
    LengthOfState* lengths;

    [[nodiscard]] HD int calculateJustDays(unsigned state, int variant) const;
};