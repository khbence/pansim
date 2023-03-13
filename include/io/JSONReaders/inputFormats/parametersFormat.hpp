#pragma once
#include <string>
#include "nlohmann/json.hpp"
#include <vector>

namespace io {
    struct Parameters {
        struct Sex {
            std::string name;
            double symptoms;
        };

        struct Age {
            int from;
            int to;
            double symptoms;
            double transmission;
        };

        struct PreCondition {
            std::string ID;
            std::string condition;
            double symptoms;
        };

        std::vector<Sex> sex;
        std::vector<Age> age;
        std::vector<PreCondition> preCondition;
    };
}