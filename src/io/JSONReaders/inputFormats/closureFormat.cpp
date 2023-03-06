#include "closureFormat.hpp"


void io::from_json(const nlohmann::json& j, ClosureRules& p) {
    j.at("rules").get_to(p.rules);
}

void io::from_json(const nlohmann::json& j, ClosureRules::Rule& p) {
    j.at("name").get_to(p.name);
    j.at("conditionType").get_to(p.conditionType);
    j.at("parameter").get_to(p.parameter);
    j.at("threshold").get_to(p.threshold);
    j.at("threshold2").get_to(p.threshold2);
    j.at("closeAfter").get_to(p.closeAfter);
    j.at("openAfter").get_to(p.openAfter);
    j.at("locationTypesToClose").get_to(p.locationTypesToClose);
}
