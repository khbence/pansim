#include "agentsFormat.hpp"

namespace io {
void from_json(const nlohmann::json& j, Agents& p) { j.at("agents").get_to(p.people); }

void from_json(const nlohmann::json& j, Agents::Person& p) {
    j.at("age").get_to(p.age);
    j.at("sex").get_to(p.sex);
    j.at("preCond").get_to(p.preCond);
    j.at("state").get_to(p.state);
    j.at("typeID").get_to(p.typeID);
    j.at("locations").get_to(p.locations);
}

void from_json(const nlohmann::json& j, Agents::Person::Location& p) {
    j.at("typeID").get_to(p.typeID);
    j.at("locID").get_to(p.locID);
}

}// namespace io