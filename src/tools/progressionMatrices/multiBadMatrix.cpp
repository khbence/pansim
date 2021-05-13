#include "multiBadMatrix.h"

MultiBadMatrix::NextStates::NextStates(unsigned _badCount,
    thrust::pair<unsigned, float>* _bad,
    unsigned _neutralCount,
    thrust::pair<unsigned, float>* _neutral)
    : badCount(_badCount), bad(_bad), neutralCount(_neutralCount), neutral(_neutral) {}

thrust::pair<unsigned, bool> HD MultiBadMatrix::NextStates::selectNext(
    float scalingSypmtons) const {
    if (neutralCount == 0) { scalingSypmtons = 1.0; }
    double random = RandomGenerator::randomUnit();
    double preSum = 0.0;
    double badSum = 0.0;
    for (unsigned i = 0; i < badCount; ++i) {
        preSum += bad[i].second * scalingSypmtons;
        if (random < preSum) { return thrust::make_pair<unsigned, bool>(bad[i].first, true); }
        badSum += bad[i].second;
    }

    double badScaling = 1.0 + ((badSum - preSum) / (1.0 - badSum));
    if (badCount == 0) { badScaling = 1.0; }

    unsigned idx = 0;
    do {
        preSum += neutral[idx].second * badScaling;
        ++idx;
    } while (preSum < random);
    idx--;
    assert(neutral[idx].first < 15);
    return thrust::make_pair<unsigned, bool>(neutral[idx].first, false);
}

void MultiBadMatrix::NextStatesInit::addBad(std::pair<unsigned, float> bad_p) {
    bad.push_back(bad_p);
}

void MultiBadMatrix::NextStatesInit::addNeutral(std::pair<unsigned, float> newNeutral) {
    neutral.push_back(newNeutral);
}

void MultiBadMatrix::NextStatesInit::cleanUp(unsigned ownIndex) {
    if (neutral.empty() && bad.empty()) { neutral.emplace_back(ownIndex, 1.0F); }
}

bool doubleIsZero(double value) { return (0.9999 < value) && (value < 1.0001); }

MultiBadMatrix::MultiBadMatrix(const parser::TransitionFormat& inputData)
    : BasicLengthAbstract(inputData.states.size()) {
    std::vector<NextStatesInit> initTransitions(inputData.states.size());

    // malloc instead of new, because this way we can use free in both CPU and
    // GPU code
    transitions = (NextStates*)malloc(sizeof(NextStates) * inputData.states.size());

    auto getStateIndex = [&inputData](const std::string& name) {
        unsigned idx = 0;
        while (inputData.states[idx].stateName != name && idx < inputData.states.size()) { ++idx; }
        if (idx == inputData.states.size()) { throw(IOProgression::WrongStateName(name)); }
        return idx;
    };

    unsigned i = 0;
    for (const auto& s : inputData.states) {
        lengths[i] = LengthOfState{ s.avgLength, s.maxlength };
        double sumChance = 0.0;
        for (const auto& t : s.progressions) {
            auto idx = getStateIndex(t.name);
            sumChance += t.chance;
            if (t.isBadProgression) {
                initTransitions[i].addBad(std::make_pair(idx, t.chance));
            } else {
                initTransitions[i].addNeutral(std::make_pair(idx, t.chance));
            }
        }
        initTransitions[i].cleanUp(i);
        if (!doubleIsZero(sumChance) && !s.progressions.empty()) {
            throw(IOProgression::BadChances(s.stateName, sumChance));
        }
        thrust::pair<unsigned, float>* bads = (thrust::pair<unsigned, float>*)malloc(
            initTransitions[i].bad.size() * sizeof(thrust::pair<unsigned, float>));
        thrust::pair<unsigned, float>* neutrals = (thrust::pair<unsigned, float>*)malloc(
            initTransitions[i].neutral.size() * sizeof(thrust::pair<unsigned, float>));

        for (int j = 0; j < initTransitions[i].neutral.size(); ++j) {
            neutrals[j] = initTransitions[i].neutral[j];
        }
        for (int j = 0; j < initTransitions[i].bad.size(); ++j) {
            bads[j] = initTransitions[i].bad[j];
        }

        transitions[i] = NextStates(
            initTransitions[i].bad.size(), bads, initTransitions[i].neutral.size(), neutrals);
        ++i;
    }
}

MultiBadMatrix::MultiBadMatrix(const std::string& fileName)
    : MultiBadMatrix(DECODE_JSON_FILE(fileName, parser::TransitionFormat)) {}

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
MultiBadMatrix* MultiBadMatrix::upload() const {
    MultiBadMatrix* tmp = (MultiBadMatrix*)malloc(sizeof(MultiBadMatrix));
    tmp->numStates = numStates;
    cudaMalloc((void**)&tmp->lengths, numStates * sizeof(LengthOfState));
    cudaMemcpy(tmp->lengths, lengths, numStates * sizeof(LengthOfState), cudaMemcpyHostToDevice);
    NextStates* tmp2 = (NextStates*)malloc(numStates * sizeof(NextStates));
    memcpy(tmp2, transitions, numStates * sizeof(NextStates));
    for (unsigned i = 0; i < numStates; i++) {
        cudaMalloc((void**)&tmp2[i].neutral,
            transitions[i].neutralCount * sizeof(thrust::pair<unsigned, float>));
        cudaMemcpy(tmp2[i].neutral,
            transitions[i].neutral,
            transitions[i].neutralCount * sizeof(thrust::pair<unsigned, float>),
            cudaMemcpyHostToDevice);

        cudaMalloc(
            (void**)&tmp2[i].bad, transitions[i].badCount * sizeof(thrust::pair<unsigned, float>));
        cudaMemcpy(tmp2[i].bad,
            transitions[i].bad,
            transitions[i].badCount * sizeof(thrust::pair<unsigned, float>),
            cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&tmp->transitions, numStates * sizeof(NextStates));
    cudaMemcpy(tmp->transitions, tmp2, numStates * sizeof(NextStates), cudaMemcpyHostToDevice);
    free(tmp2);
    MultiBadMatrix* dev;
    cudaMalloc((void**)&dev, sizeof(MultiBadMatrix));
    cudaMemcpy(dev, tmp, sizeof(MultiBadMatrix), cudaMemcpyHostToDevice);
    free(tmp);
    return dev;
}
#endif

thrust::tuple<unsigned, int, bool> HD MultiBadMatrix::calculateNextState(unsigned currentState,
    float scalingSymptons) const {
    thrust::pair<unsigned, bool> ret = transitions[currentState].selectNext(scalingSymptons);
    unsigned nextState = ret.first;
    int days = lengths[nextState].calculateDays();
    return thrust::make_tuple<unsigned, int, bool>(nextState, days, ret.second);
}