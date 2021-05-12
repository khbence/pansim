#pragma once
#include "basicLengthAbstract.h"
#include <vector>
#include <tuple>
#include "progressionMatrixFormat.h"

class MultiBadMatrix : public BasicLengthAbstract {
    class NextStates {
    public:
        unsigned neutralCount;
        unsigned badCount;
        thrust::pair<unsigned, float>* neutral;
        thrust::pair<unsigned, float>* bad;
        NextStates(unsigned _badCount,
            thrust::pair<unsigned, float>* _bad,
            unsigned _neutralCount,
            thrust::pair<unsigned, float>* _neutral);

        [[nodiscard]] thrust::pair<unsigned, bool> HD selectNext(float scalingSypmtons) const;
    };

    class NextStatesInit {
    public:
        // pair<index of new state,  raw chance to get there>
        std::vector<std::pair<unsigned, float>> bad;
        std::vector<std::pair<unsigned, float>> neutral;

        NextStatesInit() = default;

        void addBad(std::pair<unsigned, float> bad_p);
        void addNeutral(std::pair<unsigned, float> newNeutral);
        void cleanUp(unsigned ownIndex);
    };

public:
    NextStates* transitions;

public:
    MultiBadMatrix() = default;

    explicit MultiBadMatrix(const parser::TransitionFormat& inputData);
    explicit MultiBadMatrix(const std::string& fileName);

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    MultiBadMatrix* upload() const;
#endif

    [[nodiscard]] thrust::tuple<unsigned, int, bool> HD calculateNextState(unsigned currentState,
        float scalingSymptons) const;
};