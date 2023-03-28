#pragma once
#include "globalStates.hpp"
#include <string>
#include "progressionMatrices.hpp"
#include "progressionMatrixFormat.hpp"
#include "agentMeta.hpp"
#include "agentsList.hpp"
#include <vector>
#include <map>
#include <array>
#include "progressionType.hpp"

using ProgressionMatrix = MultiBadMatrix;

class DynamicPPState {
    float infectious = 0.0;
    unsigned progressionID = 0;

    char state = 0;// a number
    short daysBeforeNextState = -1;
    std::array<float, globalConstants::MAX_STRAINS> float susceptible = {0.0};
    uint8_t variant = 0;

    static HD ProgressionMatrix& getTransition(unsigned progressionID_p);

    void HD updateMeta();

public:
    static std::string initTransitionMatrix(
        std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>& inputData,
        parser::ProgressionDirectory& config,
        std::vector<float> &variantMultiplier);
    static HD unsigned getNumberOfStates();
    static std::vector<std::string> getStateNames();

    DynamicPPState();
    DynamicPPState(const std::string& name, unsigned progressionID_p);
    void HD gotInfected(uint8_t variant);
    bool HD update(float scalingSymptons, AgentStats& agentStats, BasicAgentMeta &meta, unsigned simTime, unsigned agentID, unsigned tracked, unsigned timeStep);
    [[nodiscard]] char HD getStateIdx() const { return state; }
    [[nodiscard]] states::WBStates HD getWBState() const;
    void HD setInfectious(float inf)  { infectious = inf; }
    void HD reduceInfectiousness(float multiplier);
    [[nodiscard]] float HD isInfectious() const { return infectious; }
    [[nodiscard]] float HD isInfectious(uint8_t variant) const { return (variant == this->variant) ? infectious : 0.0f; }
    [[nodiscard]] float HD getSusceptible(uint8_t variant) const { return susceptible[variant]; }
    [[nodiscard]] uint8_t HD getVariant() const { return variant; }
    void HD setSusceptible(float s, uint8_t variant) { this->susceptible[variant] = s; }
    [[nodiscard]] bool HD isInfected() const;
    [[nodiscard]] char HD die(bool covid);
    [[nodiscard]] float HD getAccuracyPCR() const;
    [[nodiscard]] float HD getAccuracyAntigen() const;
};
