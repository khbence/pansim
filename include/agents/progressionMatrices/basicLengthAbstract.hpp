#pragma once
#include "randomGenerator.h"

class BasicLengthAbstract {
protected:
    class LengthOfState {
        float avgLength[MAX_STRAINS];
        float maxLength[MAX_STRAINS];
        double p[MAX_STRAINS];

        [[nodiscard]] static double expectedLength(double p, float max);
        [[nodiscard]] static double calculateModifiedP(double p, float avgLength, float maxLength);

    public:
        LengthOfState() = default;
        LengthOfState(std::vector<float> avgLength_p, std::vector<float> maxLength_p);
        [[nodiscard]] HD int calculateDays(int variant) const;
    };

    BasicLengthAbstract(std::size_t n);

public:
    LengthOfState* lengths;
    unsigned numStates;

    [[nodiscard]] HD int calculateJustDays(unsigned state, int variant) const;
};