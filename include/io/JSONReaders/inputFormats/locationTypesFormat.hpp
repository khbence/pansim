#pragma once
#include <string>
#include "nlohmann/json.hpp"
#include <vector>

namespace io {
    struct LocationTypes {
        struct Type {
            int ID;
            std::string name;
        };

        unsigned publicSpace, home, hospital, doctor, school, classRoom, work, nurseryHome;
        std::vector<Type> types;
    };

    void from_json(const nlohmann::json& j, LocationTypes& p);
    void from_json(const nlohmann::json& j, LocationTypes::Type& p);
};