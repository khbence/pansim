#include "locationTypesFormat.hpp"

void io::from_json(const nlohmann::json& j, io::LocationTypes& p) {
    j.at("publicSpace").get_to(p.publicSpace);
    j.at("home").get_to(p.home);
    j.at("hospital").get_to(p.hospital);
    j.at("doctor").get_to(p.doctor);
    j.at("school").get_to(p.school);
    j.at("classroom").get_to(p.classRoom);
    j.at("work").get_to(p.work);
    j.at("nurseryhome").get_to(p.nurseryHome);
    j.at("types").get_to(p.types);
}

void io::from_json(const nlohmann::json& j, io::LocationTypes::Type& p) {
    j.at("ID").get_to(p.ID);
    j.at("name").get_to(p.name);
}