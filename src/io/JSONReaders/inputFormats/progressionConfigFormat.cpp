#include "progressionConfigFormat.hpp"

namespace io {
void from_json(const nlohmann::json& j, ProgressionDirectory::StateInformation& stateInformation) {
    j.at("stateNames").get_to(stateInformation.stateNames);
    j.at("firstInfectedState").get_to(stateInformation.firstInfectedState);
    j.at("nonCOVIDDeadState").get_to(stateInformation.nonCOVIDDeadState);
    j.at("susceptibleStates").get_to(stateInformation.susceptibleStates);
    j.at("infectedStates").get_to(stateInformation.infectedStates);
}

void from_json(const nlohmann::json& j, ProgressionDirectory::ProgressionFile& progressionFile) {
    j.at("fileName").get_to(progressionFile.fileName);
    j.at("age").get_to(progressionFile.age);
    j.at("preCond").get_to(progressionFile.preCond);
}

void from_json(const nlohmann::json& j, ProgressionDirectory::SingleState& singleState) {
    j.at("stateName").get_to(singleState.stateName);
    j.at("WB").get_to(singleState.WB);
    j.at("infectious").get_to(singleState.infectious);
    j.at("accuracyPCR").get_to(singleState.accuracyPCR);
    j.at("accuracyAntigen").get_to(singleState.accuracyAntigen);
}

void from_json(const nlohmann::json& j, ProgressionDirectory& progressionDirectory) {
    j.at("stateInformation").get_to(progressionDirectory.stateInformation);
    j.at("transitionMatrices").get_to(progressionDirectory.transitionMatrices);
    j.at("states").get_to(progressionDirectory.states);
};
}