#include "locationsFormat.hpp"


void io::from_json(const nlohmann::json& j, Locations& p) {
    j.at("places").get_to(p.places);
}

void io::from_json(const nlohmann::json& j, Locations::Place& p) {
    j.at("ID").get_to(p.ID);
    j.at("state").get_to(p.state);
    j.at("type").get_to(p.type);
    j.at("essential").get_to(p.essential);
    j.at("area").get_to(p.area);
    j.at("capacity").get_to(p.capacity);
    j.at("coordinates").get_to(p.coordinates);
    j.at("ageInter").get_to(p.ageInter);
}
