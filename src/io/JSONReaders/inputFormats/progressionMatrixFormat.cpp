#include "progressionMatrixFormat.hpp"

namespace io {
void from_json(const nlohmann::json& j, TransitionFormat::SingleState::Progression& progression) {
    j.at("name").get_to(progression.name);
    j.at("chance").get_to(progression.chance);
    j.at("isBadProgression").get_to(progression.isBadProgression);
}

void from_json(const nlohmann::json& j, TransitionFormat::SingleState& singleState) {
    j.at("stateName").get_to(singleState.stateName);
    j.at("avgLength").get_to(singleState.avgLength);
    j.at("maxlength").get_to(singleState.maxlength);
    j.at("progressions").get_to(singleState.progressions);
}

void from_json(const nlohmann::json& j, TransitionFormat& transitionFormat) {
    j.at("states").get_to(transitionFormat.states);
}
}