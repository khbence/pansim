#pragma once
#include <string>
#include "nlohmann/json.hpp"
#include <vector>
#include <array>

namespace io {
    struct Locations {
        struct Place {
            std::string ID, state;
            int type, essential, area, capacity;
            std::vector<double> coordinates;
            std::array<int, 2> ageInter;
        };

        std::vector<Place> places;
    };

    void from_json(const nlohmann::json& j, Locations& p);
    void from_json(const nlohmann::json& j, Locations::Place& p);
};