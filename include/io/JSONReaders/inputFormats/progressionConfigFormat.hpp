#pragma once
#include "nlohmann/json.hpp"
#include <vector>
#include <string>

namespace io {
struct ProgressionDirectory {
    struct StateInformation {
        std::vector<std::string> stateNames;
        std::string firstInfectedState;
        std::string nonCOVIDDeadState;
        std::vector<std::string> susceptibleStates;
        std::vector<std::string> infectedStates;
    };

    struct ProgressionFile {
        std::string fileName;
        std::vector<int> age;
        std::string preCond;
    };

    struct SingleState {
        std::string stateName;
        std::string WB;
        float infectious;
        float accuracyPCR;
        float accuracyAntigen;
    };

    StateInformation stateInformation;
    std::vector<ProgressionFile> transitionMatrices;
    std::vector<SingleState> states;
};

void from_json(const nlohmann::json& j, ProgressionDirectory::StateInformation& stateInformation);
void from_json(const nlohmann::json& j, ProgressionDirectory::ProgressionFile& progressionFile);
void from_json(const nlohmann::json& j, ProgressionDirectory::SingleState& singleState);
void from_json(const nlohmann::json& j, ProgressionDirectory& progressionDirectory);
}// namespace io