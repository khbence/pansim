#pragma once
#include <string>
#include "nlohmann/json.hpp"
#include <vector>

namespace io {
    struct ClosureRules {
        struct Rule {
            std::string name, conditionType, parameter;
            double threshold, threshold2;
            int closeAfter, openAfter;
            std::vector<unsigned> locationTypesToClose;
        };

        std::vector<Rule> rules;
    };

    void from_json(const nlohmann::json& j, ClosureRules& p);
    void from_json(const nlohmann::json& j, ClosureRules::Rule& p);
}