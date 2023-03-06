#pragma once
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

namespace io {
struct ConfigRandom {
    // used in many places here
    struct Switch {
        std::string value;
        long double chance;
    };

    struct IrregularChances {
        struct Detail {
            std::string value;
            long double chanceForType;
            long double chanceFromAllIrregulars;
            std::vector<Switch> switchedToWhat;
        };

        double generalChance;
        std::vector<Detail> detailsOfChances;
    };

    struct StatesForAge {
        struct Distribution : public Switch {
            long double diagnosedChance;
        };

        unsigned char ageStart, ageEnd;
        std::vector<Distribution> distribution;
    };

    IrregularChances irregularLocationChance;
    std::vector<Switch> locationTypeDistribution;
    std::vector<Switch> preCondDistribution;
    std::vector<StatesForAge> stateDistribution;
    std::vector<Switch> agentTypeDistribution;
};

void from_json(const nlohmann::json& j, ConfigRandom& p);
void from_json(const nlohmann::json& j, ConfigRandom::Switch& p);
void from_json(const nlohmann::json& j, ConfigRandom::IrregularChances& p);
void from_json(const nlohmann::json& j, ConfigRandom::IrregularChances::Detail& p);
void from_json(const nlohmann::json& j, ConfigRandom::StatesForAge& p);
void from_json(const nlohmann::json& j, ConfigRandom::StatesForAge::Distribution& p);
}// namespace io