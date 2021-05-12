#pragma once
#include <map>
#include <array>
#include <utility>
#include <vector>
#include <string>
#include "parametersFormat.h"
#include "datatypes.h"

class BasicAgentMeta {
    class AgeInterval {
        unsigned from;
        unsigned to;
        float symptoms;
        float transmission;

    public:
        explicit AgeInterval(parser::Parameters::Age in);
        bool operator==(unsigned age) const { return (from <= age) && (age < to); }
        [[nodiscard]] float getSymptoms() const;
        [[nodiscard]] float getTransmission() const;
    };

    float scalingSymptoms = 1.0;
    float scalingTransmission = 1.0;
    uint8_t age = 0;
    uint8_t preCondIdx = 0;
    bool sex = false;

    static std::array<std::pair<char, float>, 2> sexScaling;
    static std::vector<AgeInterval> ageScaling;
    static std::map<std::string, float> preConditionScaling;

public:
    static void initData(const parser::Parameters& inputData);

    BasicAgentMeta(char gender, unsigned age, std::string preCondition);
    BasicAgentMeta();

    [[nodiscard]] float HD getScalingSymptoms() const;
    [[nodiscard]] float HD getScalingTransmission() const;

    [[nodiscard]] uint8_t HD getAge() const;
    [[nodiscard]] bool HD getSex() const;
    [[nodiscard]] uint8_t HD getPrecondIdx() const;
};