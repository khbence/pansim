#include "agentsFormat.hpp"
#include "agentsTypeFormat.hpp"

namespace io {
void from_json(const nlohmann::json& j, AgentTypes& p) {
    j.at("types").get_to(p.types);
}

void from_json(const nlohmann::json& j, AgentTypes::Type& p) {
    j.at("name").get_to(p.name);
    j.at("ID").get_to(p.ID);
    j.at("schedule").get_to(p.schedule);
}

void from_json(const nlohmann::json& j, AgentTypes::Type::Schedule& p) {
    j.at("type").get_to(p.type);
    j.at("chance").get_to(p.chance);
    j.at("start").get_to(p.start);
    j.at("end").get_to(p.end);
    j.at("duration").get_to(p.duration);
}
}// namespace io