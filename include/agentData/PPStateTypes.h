#pragma once
#include "globalStates.h"
#include "progressionMatrices.h"
#include <string>


class PPStateSIRAbstract {
protected:
    states::SIRD state;

public:
    static HD unsigned getNumberOfStates() { return 0; };

    static void initTransitionMatrix(const std::string& inputFile) {}

    HD explicit PPStateSIRAbstract(states::SIRD s);
    states::SIRD parseState(const std::string& input);
    virtual HD void update(float scalingSymptons) = 0;
    virtual HD void gotInfected();
    [[nodiscard]] HD states::SIRD getSIRD() const;
    [[nodiscard]] HD states::WBStates getWBState() const;
    virtual HD char getStateIdx() const = 0;
    [[nodiscard]] virtual bool HD isInfectious() const { return state == states::SIRD::I; }
    [[nodiscard]] virtual float HD getSusceptible() const { return state == states::SIRD::S; }
};

class PPStateSIRBasic : public PPStateSIRAbstract {
public:
    static HD unsigned getNumberOfStates() { return 4; };
    PPStateSIRBasic();
    explicit PPStateSIRBasic(states::SIRD s);
    void HD update(float scalingSymptons) override;
};

class PPStateSIRextended : public PPStateSIRAbstract {
    char subState = 0;// I1, I2, I3 ... R1, R2
    char idx = 0;

    // -1 it will remain in that state until something special event happens,
    // like got infected -2 has to be calculated during update
    int daysBeforeNextState = -1;

private:
    HD SingleBadTransitionMatrix& getTransition();
    HD unsigned* getStartingIdx();

    HD void applyNewIdx();

    static void printHeader();

public:
    static HD unsigned getNumberOfStates();
    HD PPStateSIRextended();
    explicit HD PPStateSIRextended(states::SIRD s);
    explicit HD PPStateSIRextended(char idx_p);
    void HD gotInfected() override;
    [[nodiscard]] char HD getSubState() { return subState; }
    static void initTransitionMatrix(const std::string& inputFile);
    void HD update(float scalingSymptons) override;
    [[nodiscard]] char HD getStateIdx() const override;
    [[nodiscard]] bool HD isInfectious() const override {
        return (state == states::SIRD::S) && (subState > 0);
    }
};
