#pragma once
#include <array>
#include <optional>
#include <vector>
#include <string>
#include "thrust/pair.h"
#include "progressionMatrixFormat.hpp"
#include "customExceptions.hpp"
#include <algorithm>
#include "randomGenerator.hpp"
#include "basicLengthAbstract.hpp"

class SingleBadTransitionMatrix : public BasicLengthAbstract {
    class NextStates {
        bool hasBad;
        thrust::pair<unsigned, float> bad;

    public:
        unsigned neutralCount;
        thrust::pair<unsigned, float>* neutral;
        NextStates(bool _hasBad,
            thrust::pair<unsigned, float> _bad,
            thrust::pair<unsigned, float>* _neutral,
            unsigned _neutralCount);

        [[nodiscard]] HD unsigned selectNext(float scalingSypmtons) const;
    };

    class NextStatesInit {
    public:
        // pair<index of new state,  raw chance to get there>
        std::optional<std::pair<unsigned, float>> bad;
        std::vector<std::pair<unsigned, float>> neutral;

        NextStatesInit() = default;

        void addBad(std::pair<unsigned, float> bad_p);
        void addNeutral(std::pair<unsigned, float> newNeutral);
        void cleanUp(unsigned ownIndex);
    };

public:
    NextStates* transitions;

public:
    SingleBadTransitionMatrix() = default;

    explicit SingleBadTransitionMatrix(const io::TransitionFormat& inputData);
    explicit SingleBadTransitionMatrix(const std::string& fileName);

    ~SingleBadTransitionMatrix();

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    SingleBadTransitionMatrix* upload() const;
#endif

    [[nodiscard]] thrust::pair<unsigned, int> HD calculateNextState(unsigned currentState, float scalingSymptons, int variant) const;
};