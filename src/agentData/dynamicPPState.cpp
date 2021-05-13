#include "dynamicPPState.h"
#include <cassert>
#include "customExceptions.h"

// static stuff
namespace detail {
    namespace DynamicPPState {
        unsigned h_numberOfStates = 0;
        char h_firstInfectedState = 0;
        char h_nonCOVIDDeadState = 0;
        char h_deadState;
        std::vector<float> h_infectious;
        std::vector<float> h_variantMultiplier;
        std::vector<float> h_accuracyPCR;
        std::vector<float> h_accuracyAntigen;
        std::vector<bool> h_susceptible;
        std::vector<bool> h_infected;
        std::vector<states::WBStates> h_WB;
        std::map<std::string, char> nameIndexMap;
        std::vector<ProgressionMatrix> h_transition;
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        __constant__ unsigned numberOfStates = 0;
        __constant__ char firstInfectedState = 0;
        __constant__ char nonCOVIDDeadState = 0;
        __constant__ float* infectious;
        __constant__ float* variantMultiplier;
        __constant__ float* accuracyPCR;
        __constant__ float* accuracyAntigen;
        __constant__ bool* susceptible;
        __constant__ bool* infected;
        __constant__ states::WBStates* WB;
        __constant__ char deadState;
        __constant__ ProgressionMatrix** d_transition;
#endif
    }// namespace DynamicPPState
}// namespace detail

HD ProgressionMatrix& DynamicPPState::getTransition(unsigned progressionID_p) {
#ifdef __CUDA_ARCH__
    return *detail::DynamicPPState::d_transition[progressionID_p];
#else
    return detail::DynamicPPState::h_transition[progressionID_p];
#endif
}

void HD DynamicPPState::updateMeta() {
#ifdef __CUDA_ARCH__
    infectious = detail::DynamicPPState::infectious[state] * detail::DynamicPPState::variantMultiplier[variant];
    susceptible = detail::DynamicPPState::susceptible[state];
#else
    infectious = detail::DynamicPPState::h_infectious[state] * detail::DynamicPPState::h_variantMultiplier[variant];
    susceptible = detail::DynamicPPState::h_susceptible[state];
#endif
}

std::string DynamicPPState::initTransitionMatrix(
    std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>&
        inputData,
    parser::ProgressionDirectory& config, float multiplier) {
    // init global parameters that are used to be static
    detail::DynamicPPState::h_numberOfStates = config.stateInformation.stateNames.size();
    detail::DynamicPPState::h_infectious =
        decltype(detail::DynamicPPState::h_infectious)(detail::DynamicPPState::h_numberOfStates);
    detail::DynamicPPState::h_variantMultiplier =
        decltype(detail::DynamicPPState::h_variantMultiplier)(2);
    detail::DynamicPPState::h_accuracyPCR =
        decltype(detail::DynamicPPState::h_accuracyPCR)(detail::DynamicPPState::h_numberOfStates);
    detail::DynamicPPState::h_accuracyAntigen =
        decltype(detail::DynamicPPState::h_accuracyAntigen)(detail::DynamicPPState::h_numberOfStates);
    detail::DynamicPPState::h_infected =
        decltype(detail::DynamicPPState::h_infected)(detail::DynamicPPState::h_numberOfStates);
    detail::DynamicPPState::h_WB =
        decltype(detail::DynamicPPState::h_WB)(detail::DynamicPPState::h_numberOfStates);
    detail::DynamicPPState::h_susceptible = decltype(detail::DynamicPPState::h_susceptible)(
        detail::DynamicPPState::h_numberOfStates, false);
    detail::DynamicPPState::h_transition.reserve(inputData.size());

    char idx = 0;
    std::string header;
    for (const auto& e : config.stateInformation.stateNames) {
        header += e + '\t';
        detail::DynamicPPState::nameIndexMap.emplace(std::make_pair(e, idx));
        ++idx;
    }

    for (const auto& e : config.states) {
        auto idx = detail::DynamicPPState::nameIndexMap.at(e.stateName);
        detail::DynamicPPState::h_WB[idx] = states::parseWBState(e.WB);
        detail::DynamicPPState::h_infectious[idx] = e.infectious;
        detail::DynamicPPState::h_accuracyPCR[idx] = e.accuracyPCR;
        detail::DynamicPPState::h_accuracyAntigen[idx] = e.accuracyAntigen;
    }

    detail::DynamicPPState::h_variantMultiplier[0] = 1.0;
    detail::DynamicPPState::h_variantMultiplier[1] = multiplier;

    for (const auto& e : config.stateInformation.infectedStates) {
        auto idx = detail::DynamicPPState::nameIndexMap.at(e);
        detail::DynamicPPState::h_infected[idx] = true;
    }

    for (const auto& e : config.stateInformation.susceptibleStates) {
        auto idx = detail::DynamicPPState::nameIndexMap.at(e);
        detail::DynamicPPState::h_susceptible[idx] = true;
    }

    detail::DynamicPPState::h_firstInfectedState =
        detail::DynamicPPState::nameIndexMap.at(config.stateInformation.firstInfectedState);
    detail::DynamicPPState::h_nonCOVIDDeadState =
        detail::DynamicPPState::nameIndexMap.at(config.stateInformation.nonCOVIDDeadState);

    for (unsigned i = 0; i < inputData.size(); ++i) {
        auto it = std::find_if(inputData.begin(), inputData.end(), [i](const auto& e) {
            return e.second.second == i;
        });
        assert(it != inputData.end());
        detail::DynamicPPState::h_transition.emplace_back(it->second.first);
    }

    //Find which state is dead due to COVID
    //TODO: have a flag so we can distinguish properly
    for (unsigned i = 0; i < detail::DynamicPPState::h_numberOfStates; i++) {
        if (detail::DynamicPPState::h_WB[i] == states::WBStates::D) {
            detail::DynamicPPState::h_deadState = i;
            break;
        }
    }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    ProgressionMatrix* tmpDevice;
    cudaMalloc((void**)&tmpDevice,
        detail::DynamicPPState::h_transition.size()
            * sizeof(decltype(detail::DynamicPPState::d_transition)));

    std::vector<ProgressionMatrix*> tmp;
    for (auto& e : detail::DynamicPPState::h_transition) { tmp.push_back(e.upload()); }
    cudaMemcpy(
        tmpDevice, tmp.data(), tmp.size() * sizeof(ProgressionMatrix*), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(
        detail::DynamicPPState::d_transition, &tmpDevice, sizeof(decltype(tmpDevice)));

    float* infTMP;
    cudaMalloc((void**)&infTMP, detail::DynamicPPState::h_numberOfStates * sizeof(float));
    cudaMemcpy(infTMP,
        detail::DynamicPPState::h_infectious.data(),
        detail::DynamicPPState::h_numberOfStates * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::infectious, &infTMP, sizeof(float*));

    float* varTMP;
    cudaMalloc((void**)&varTMP, 2 * sizeof(float));
    cudaMemcpy(varTMP,
        detail::DynamicPPState::h_variantMultiplier.data(),
        2 * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::variantMultiplier, &varTMP, sizeof(float*));

    float* accuracyPCRTMP;
    cudaMalloc((void**)&accuracyPCRTMP, detail::DynamicPPState::h_numberOfStates * sizeof(float));
    cudaMemcpy(accuracyPCRTMP,
        detail::DynamicPPState::h_accuracyPCR.data(),
        detail::DynamicPPState::h_numberOfStates * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::accuracyPCR, &accuracyPCRTMP, sizeof(float*));

    float* accuracyAntigenTMP;
    cudaMalloc((void**)&accuracyAntigenTMP, detail::DynamicPPState::h_numberOfStates * sizeof(float));
    cudaMemcpy(accuracyAntigenTMP,
        detail::DynamicPPState::h_accuracyAntigen.data(),
        detail::DynamicPPState::h_numberOfStates * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::accuracyAntigen, &accuracyAntigenTMP, sizeof(float*));

    states::SIRD* wbTMP;
    cudaMalloc((void**)&wbTMP, detail::DynamicPPState::h_numberOfStates * sizeof(states::SIRD));
    cudaMemcpy(wbTMP,
        detail::DynamicPPState::h_WB.data(),
        detail::DynamicPPState::h_numberOfStates * sizeof(states::SIRD),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::WB, &wbTMP, sizeof(states::SIRD*));

    bool* tmpSusceptible = new bool[detail::DynamicPPState::h_susceptible.size()];
    std::copy(detail::DynamicPPState::h_susceptible.begin(),
        detail::DynamicPPState::h_susceptible.end(),
        tmpSusceptible);
    bool* susTMP;
    cudaMalloc((void**)&susTMP, detail::DynamicPPState::h_numberOfStates * sizeof(bool));
    cudaMemcpy(susTMP,
        tmpSusceptible,
        detail::DynamicPPState::h_numberOfStates * sizeof(bool),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::susceptible, &susTMP, sizeof(bool*));

    bool* tmpInfected = new bool[detail::DynamicPPState::h_infected.size()];
    std::copy(detail::DynamicPPState::h_infected.begin(),
        detail::DynamicPPState::h_infected.end(),
        tmpInfected);

    cudaMalloc((void**)&infTMP, detail::DynamicPPState::h_infected.size() * sizeof(bool));
    cudaMemcpy(infTMP,
        tmpInfected,
        detail::DynamicPPState::h_infected.size() * sizeof(bool),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::infected, &infTMP, sizeof(bool*));

    cudaMemcpyToSymbol(detail::DynamicPPState::firstInfectedState,
        &detail::DynamicPPState::h_firstInfectedState,
        sizeof(detail::DynamicPPState::h_firstInfectedState));
    cudaMemcpyToSymbol(detail::DynamicPPState::nonCOVIDDeadState,
        &detail::DynamicPPState::h_nonCOVIDDeadState,
        sizeof(detail::DynamicPPState::h_nonCOVIDDeadState));
    cudaMemcpyToSymbol(detail::DynamicPPState::deadState,
        &detail::DynamicPPState::h_deadState,
        sizeof(detail::DynamicPPState::h_deadState));
#endif
    return header;
}

HD unsigned DynamicPPState::getNumberOfStates() {
#ifdef __CUDA_ARCH__
    return detail::DynamicPPState::numberOfStates;
#else
    return detail::DynamicPPState::h_numberOfStates;
#endif
}

std::vector<std::string> DynamicPPState::getStateNames() {
    std::vector<std::string> names(detail::DynamicPPState::h_numberOfStates);
    for (const auto& e : detail::DynamicPPState::nameIndexMap) { names[e.second] = e.first; }
    return names;
}

DynamicPPState::DynamicPPState(const std::string& name, unsigned progressionID_p)
    : progressionID(progressionID_p),
      state(detail::DynamicPPState::nameIndexMap.find(name)->second),
      daysBeforeNextState(getTransition(progressionID).calculateJustDays(state)) {
    updateMeta();
}

void HD DynamicPPState::gotInfected(uint8_t v) {
#ifdef __CUDA_ARCH__
    state = detail::DynamicPPState::firstInfectedState;
#else
    state = detail::DynamicPPState::h_firstInfectedState;
#endif
    variant = v;
    daysBeforeNextState = -2;
    updateMeta();
}

bool HD DynamicPPState::update(float scalingSymptons,
    AgentStats& stats,
    unsigned simTime,
    unsigned agentID,
    unsigned tracked) {
    if (daysBeforeNextState == -2) {
        daysBeforeNextState = getTransition(progressionID).calculateJustDays(state);
    }
    else if (daysBeforeNextState > 0) { --daysBeforeNextState; return false; } //Do not subtract if just infected

    if (daysBeforeNextState == 0) {
        states::WBStates oldWBState = this->getWBState();
        auto oldState = state;
        auto tmp = getTransition(progressionID).calculateNextState(state, scalingSymptons);
        state = thrust::get<0>(tmp);
        updateMeta();
        daysBeforeNextState = thrust::get<1>(tmp);

        if (thrust::get<2>(tmp)) {// was a bad progression
            stats.worstState = state;
            stats.worstStateTimestamp = simTime;
            if (agentID == tracked) {
                printf(
                    "Agent %d bad progression %d->%d WBState: %d->%d for %d "
                    "days\n",
                    agentID,
                    oldState,
                    state,
                    oldWBState,
                    this->getWBState(),
                    daysBeforeNextState);
            }
        } else if (oldState != state) {// if (oldWBState != states::WBStates::W) this will record any
                // good progression!
            stats.worstStateEndTimestamp = simTime;
            if (agentID == tracked) {
                printf("Agent %d good progression %d->%d WBState: %d->%d\n",
                    agentID,
                    oldState,
                    state,
                    oldWBState,
                    this->getWBState());
            }
            if (!this->isInfected()) {
                return true;// recovered
            }
        }
    }
    return false;
}

states::WBStates DynamicPPState::getWBState() const {
#ifdef __CUDA_ARCH__
    return detail::DynamicPPState::WB[state];
#else
    return detail::DynamicPPState::h_WB[state];
#endif
}

HD char DynamicPPState::die(bool covid) {
    daysBeforeNextState = -1;
#ifdef __CUDA_ARCH__
    state = covid ? detail::DynamicPPState::deadState : detail::DynamicPPState::nonCOVIDDeadState; 
    updateMeta();
    return state;
#else
    state = covid ? detail::DynamicPPState::h_deadState : detail::DynamicPPState::h_nonCOVIDDeadState; ; 
    updateMeta();
    return state;
#endif
}
bool HD DynamicPPState::isInfected() const {
#ifdef __CUDA_ARCH__
    return detail::DynamicPPState::infected[state];
#else
    return detail::DynamicPPState::h_infected[state];
#endif
}


float HD DynamicPPState::getAccuracyPCR() const {
#ifdef __CUDA_ARCH__
    return detail::DynamicPPState::accuracyPCR[state];
#else
    return detail::DynamicPPState::h_accuracyPCR[state];
#endif
}

float HD DynamicPPState::getAccuracyAntigen() const {
#ifdef __CUDA_ARCH__
    return detail::DynamicPPState::accuracyAntigen[state];
#else
    return detail::DynamicPPState::h_accuracyAntigen[state];
#endif
}

