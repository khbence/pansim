#include "singleBadMatrix.h"

SingleBadTransitionMatrix::NextStates::NextStates(bool _hasBad,
    thrust::pair<unsigned, float> _bad,
    thrust::pair<unsigned, float>* _neutral,
    unsigned _neutralCount)
    : hasBad(_hasBad), bad(_bad), neutral(_neutral), neutralCount(_neutralCount) {}

HD unsigned SingleBadTransitionMatrix::NextStates::selectNext(float scalingSypmtons) const {
    double random = RandomGenerator::randomUnit();
    double iterator = 0.0;
    double remainders = 0.0;
    if (hasBad) {
        iterator = bad.second * scalingSypmtons;
        if (random < iterator) { return bad.first; }
        remainders = (bad.second - iterator) / neutralCount;
    }
    unsigned idx = 0;
    do {
        iterator += neutral[idx].second + remainders;
        ++idx;
    } while (iterator < random);
    idx--;
    return neutral[idx].first;
}

void SingleBadTransitionMatrix::NextStatesInit::addBad(std::pair<unsigned, float> bad_p) {
    if (bad) { throw(IOProgression::TooMuchBad(bad_p.first)); }
    bad = bad_p;
}

void SingleBadTransitionMatrix::NextStatesInit::addNeutral(std::pair<unsigned, float> newNeutral) {
    neutral.push_back(newNeutral);
}

void SingleBadTransitionMatrix::NextStatesInit::cleanUp(unsigned ownIndex) {
    if (neutral.empty()) {
        if (bad) {
            neutral.push_back(bad.value());
            bad.reset();
        } else {
            neutral.emplace_back(ownIndex, 1.0F);
        }
    }
}

SingleBadTransitionMatrix::SingleBadTransitionMatrix(const parser::TransitionFormat& inputData)
    : BasicLengthAbstract(inputData.states.size()) {
    std::vector<NextStatesInit> initTransitions(inputData.states.size());
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
        if (sumChance != 1.0 && !s.progressions.empty()) {
            throw(IOProgression::BadChances(s.stateName, sumChance));
        }
        thrust::pair<unsigned, float> badVal = initTransitions[i].bad
                                                   ? initTransitions[i].bad.value()
                                                   : thrust::pair<unsigned, float>(0, 0.0f);
        thrust::pair<unsigned, float>* neutrals = (thrust::pair<unsigned, float>*)malloc(
            initTransitions[i].neutral.size() * sizeof(thrust::pair<unsigned, float>));
        for (int j = 0; j < initTransitions[i].neutral.size(); j++)
            neutrals[j] = initTransitions[i].neutral[j];
        transitions[i] = NextStates(initTransitions[i].bad ? true : false,
            badVal,
            neutrals,
            initTransitions[i].neutral.size());
        ++i;
    }
}

SingleBadTransitionMatrix::SingleBadTransitionMatrix(const std::string& fileName)
    : SingleBadTransitionMatrix(DECODE_JSON_FILE(fileName, parser::TransitionFormat)) {}

SingleBadTransitionMatrix::~SingleBadTransitionMatrix() {
    for (unsigned i = 0; i < numStates; i++) { free(transitions[i].neutral); }
    free(transitions);
    free(lengths);
}

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
SingleBadTransitionMatrix* SingleBadTransitionMatrix::upload() const {
    SingleBadTransitionMatrix* tmp =
        (SingleBadTransitionMatrix*)malloc(sizeof(SingleBadTransitionMatrix));
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
    }
    cudaMalloc((void**)&tmp->transitions, numStates * sizeof(NextStates));
    cudaMemcpy(tmp->transitions, tmp2, numStates * sizeof(NextStates), cudaMemcpyHostToDevice);
    free(tmp2);
    SingleBadTransitionMatrix* dev;
    cudaMalloc((void**)&dev, sizeof(SingleBadTransitionMatrix));
    cudaMemcpy(dev, tmp, sizeof(SingleBadTransitionMatrix), cudaMemcpyHostToDevice);
    free(tmp);
    return dev;
}
#endif

thrust::pair<unsigned, int> HD SingleBadTransitionMatrix::calculateNextState(unsigned currentState,
    float scalingSymptons) const {
    unsigned nextState = transitions[currentState].selectNext(scalingSymptons);
    int days = lengths[nextState].calculateDays();
    return thrust::make_pair(nextState, days);
}