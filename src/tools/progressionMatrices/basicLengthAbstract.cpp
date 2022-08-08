#include "basicLengthAbstract.h"
#include "smallTools.h"

double BasicLengthAbstract::LengthOfState::expectedLength(double pInput, float max) {
    if (pInput < 0.0) { return 0.0; }
    double sumExpected = 0.0;
    double sumChance = 0.0;
    for (unsigned i = 0; i < (unsigned)round(max); ++i) {
        double chance = std::pow(1 - pInput, i) * pInput;
        sumExpected += (i + 1) * chance;
        sumChance += chance;
    }
    return sumExpected * (1.0 / sumChance);
}

double BasicLengthAbstract::LengthOfState::calculateModifiedP(double p, float avgLength, float maxLength) {
    if (avgLength == -1.0f) { return 1.0; }
    double n = (double)avgLength;
    double pNew = 1.0 / n;
    float maxLocal = maxLength;
    auto objFunction = [maxLocal, n](double pCurrent) -> double { return expectedLength(pCurrent, maxLocal) - n; };
    try {
        pNew = SecantMethod(objFunction, pNew, pNew - (pNew * 0.1), 0.0001);
    } catch (const std::runtime_error& e) { std::cerr << e.what() << " original p value will be used instead \n"; }
    // To check if an extremely small p value
    if ((pNew * 10.0 < (1.0 / n)) || pNew > 1.0) { pNew = 1.0 / n; }
    //    std::cout << "n: " << avgLength  << " max: " << maxLocal << " old p: " << 1.0/n << " new p: " << pNew << '\n';
    return pNew;
}

BasicLengthAbstract::LengthOfState::LengthOfState(std::vector<float> avgLength_p, std::vector<float> maxLength_p) {
    for (int v = 0; v < avgLength_p.size(); v++) {
        p[v] = 1.0f/avgLength_p[v];
        maxLength[v] = maxLength_p[v];
        avgLength[v] = avgLength_p[v];
        if (maxLength[v] == -1.0f) { maxLength[v] = std::numeric_limits<float>::max(); }
        p[v] = calculateModifiedP(p[v], avgLength[v], maxLength[v]);
    }
    for (int v = avgLength_p.size(); v < MAX_STRAINS; v++) {
        maxLength[v] = maxLength[avgLength_p.size()-1];
        avgLength[v] = avgLength[avgLength_p.size()-1];
        p[v] = p[avgLength_p.size()-1];
    }
}

// Note: [0, maxLength), because the 0 will run for a day, so the maxLength
// would run for maxLength+1 days
[[nodiscard]] HD int BasicLengthAbstract::LengthOfState::calculateDays(int variant) const {
    if (avgLength[variant] == -1.0) { return -1; }
    int days = RandomGenerator::geometric(p[variant]);
    while (maxLength[variant] < days) { days = RandomGenerator::geometric(p[variant]); }
    return days;
}

BasicLengthAbstract::BasicLengthAbstract(std::size_t n)
    : numStates(n), lengths((LengthOfState*)malloc(sizeof(LengthOfState) * n)) {}

HD int BasicLengthAbstract::calculateJustDays(unsigned state, int variant) const { return lengths[state].calculateDays(variant); }
