#pragma once
#include "nlohmann/json.hpp"
#include <vector>
#include <string>

namespace io {
struct TransitionFormat {
    struct SingleState {
        struct Progression {
            std::string name;
            double chance;
            bool isBadProgression;
        };

        std::string stateName;
        std::vector<float> avgLength;
        std::vector<float> maxlength;
        std::vector<Progression> progressions;
    };

    std::vector<::io::TransitionFormat::SingleState> states;
};

void from_json(const nlohmann::json& j, TransitionFormat::SingleState::Progression& progression);
void from_json(const nlohmann::json& j, TransitionFormat::SingleState& singleState);
void from_json(const nlohmann::json& j, TransitionFormat& transition);
}