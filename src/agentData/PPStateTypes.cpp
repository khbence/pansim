#include "PPStateTypes.h"
#include "customExceptions.h"

// static stuff
namespace detail {
    namespace PPStateSIRextended {
        __device__ unsigned numberOfStates = 1 + 6 + 3 + 1;// S + I + R + D
        unsigned h_numberOfStates = 1 + 6 + 3 + 1;// S + I + R + D
        __device__ unsigned startingIdx[5] = { 0, 1, 7, 10, 11 };// to convert from idx to state
        unsigned h_startingIdx[5] = { 0, 1, 7, 10, 11 };
        SingleBadTransitionMatrix* transition;
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        __device__ SingleBadTransitionMatrix* transition_gpu;
#endif
    }// namespace PPStateSIRextended
}// namespace detail

// Abstract
HD PPStateSIRAbstract::PPStateSIRAbstract(states::SIRD s) : state(s) {}

states::SIRD PPStateSIRAbstract::parseState(const std::string& input) {
    if (input.length() != 1) { throw IOAgents::InvalidPPState(input); }
    char s = static_cast<char>(std::toupper(input.front()));
    switch (s) {
    case 'S':
        return states::SIRD::S;
    case 'I':
        return states::SIRD::I;
    case 'R':
        return states::SIRD::R;
    case 'D':
        return states::SIRD::D;
    default:
        throw IOAgents::InvalidPPState(input);
    }
}

void PPStateSIRAbstract::gotInfected() { this->state = states::SIRD::I; }

[[nodiscard]] HD states::SIRD PPStateSIRAbstract::getSIRD() const { return state; }

[[nodiscard]] HD states::WBStates PPStateSIRAbstract::getWBState() const {
    switch (state) {
    case states::SIRD::R:
    case states::SIRD::S:
        return states::WBStates::W;
    case states::SIRD::I:
        return states::WBStates::N;
    case states::SIRD::D:
        return states::WBStates::D;
    default:
        return states::WBStates::W;
    }
}

// Basic
// TODO

// Extended
// PPStateSIRextended::SingleBadTransitionMatrix<PPStateSIRextended::numberOfStates>
// transition;

HD void PPStateSIRextended::applyNewIdx() {
    state = states::SIRD::S;
    for (int i = 0; i < 4; i++) {
        if (idx >= getStartingIdx()[i] && idx < getStartingIdx()[i + 1]) {
            state = (states::SIRD)i;
            subState = idx - getStartingIdx()[i];
        }
    }
}


HD SingleBadTransitionMatrix& PPStateSIRextended::getTransition() {
#ifdef __CUDA_ARCH__
    return *detail::PPStateSIRextended::transition_gpu;
#else
    return *detail::PPStateSIRextended::transition;
#endif
};

HD unsigned* PPStateSIRextended::getStartingIdx() {
#ifdef __CUDA_ARCH__
    return detail::PPStateSIRextended::startingIdx;
#else
    return detail::PPStateSIRextended::h_startingIdx;
#endif
}

void PPStateSIRextended::printHeader() {
    // I was lazy to do it properly
    std::cout << "S, I1, I2, I3, I4, I5, I6, R1, R2, R3, D\n";
}

HD PPStateSIRextended::PPStateSIRextended() : PPStateSIRAbstract(states::SIRD::S) {}
HD PPStateSIRextended::PPStateSIRextended(states::SIRD s) : PPStateSIRAbstract(s) {
    idx = static_cast<char>(state);
    daysBeforeNextState = getTransition().calculateJustDays(idx);
}

HD PPStateSIRextended::PPStateSIRextended(char idx_p)
    : PPStateSIRAbstract(states::SIRD::S), idx(idx_p) {
    applyNewIdx();
    daysBeforeNextState = getTransition().calculateJustDays(idx);
}

HD void PPStateSIRextended::gotInfected() {
    idx = 1;
    applyNewIdx();
    daysBeforeNextState = -2;
    // std::cout << "From " << 0 << " -> " << (int)idx<<"\n";
}

HD void PPStateSIRextended::update(float scalingSymptons) {
    // the order of the first two is intentional
    if (daysBeforeNextState == -2) { daysBeforeNextState = getTransition().calculateJustDays(idx); }
    if (daysBeforeNextState > 0) { --daysBeforeNextState; }
    if (daysBeforeNextState == 0) {
        auto tmp = getTransition().calculateNextState(idx, scalingSymptons);
        auto stateIdx = tmp.first;
        auto days = tmp.second;
        daysBeforeNextState = days;
        idx = stateIdx;
        applyNewIdx();
    }
}

void PPStateSIRextended::initTransitionMatrix(const std::string& inputFile) {
    detail::PPStateSIRextended::transition = new SingleBadTransitionMatrix(inputFile);
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    SingleBadTransitionMatrix* tmp = detail::PPStateSIRextended::transition->upload();
    cudaMemcpyToSymbol(
        detail::PPStateSIRextended::transition_gpu, &tmp, sizeof(SingleBadTransitionMatrix*));
#endif
    printHeader();
}

HD unsigned PPStateSIRextended::getNumberOfStates() {
#ifdef __CUDA_ARCH__
    return detail::PPStateSIRextended::numberOfStates;
#else
    return detail::PPStateSIRextended::h_numberOfStates;
#endif
}

HD char PPStateSIRextended::getStateIdx() const { return idx; }
