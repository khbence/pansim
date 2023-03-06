#include "configRandomFormat.hpp"


void io::from_json(const nlohmann::json& j, ConfigRandom& p) {
    j.at("irregularLocationChance").get_to(p.irregularLocationChance);
    j.at("locationTypeDistribution").get_to(p.locationTypeDistribution);
    j.at("preCondDistribution").get_to(p.preCondDistribution);
    j.at("stateDistribution").get_to(p.stateDistribution);
    j.at("agentTypeDistribution").get_to(p.agentTypeDistribution);
}

void io::from_json(const nlohmann::json& j, ConfigRandom::Switch& p) {
    j.at("value").get_to(p.value);
    j.at("chance").get_to(p.chance);
}

void io::from_json(const nlohmann::json& j, ConfigRandom::IrregularChances& p) {
    j.at("generalChance").get_to(p.generalChance);
    j.at("detailsOfChances").get_to(p.detailsOfChances);
}

void io::from_json(const nlohmann::json& j, ConfigRandom::IrregularChances::Detail& p) {
    j.at("value").get_to(p.value);
    j.at("chanceForType").get_to(p.chanceForType);
    j.at("chanceFromAllIrregulars").get_to(p.chanceFromAllIrregulars);
    j.at("switchedToWhat").get_to(p.switchedToWhat);
}

void io::from_json(const nlohmann::json& j, ConfigRandom::StatesForAge& p) {
    j.at("ageStart").get_to(p.ageStart);
    j.at("ageEnd").get_to(p.ageEnd);
    j.at("distribution").get_to(p.distribution);
}

void io::from_json(const nlohmann::json& j, ConfigRandom::StatesForAge::Distribution& p) {
    j.at("diagnosedChance").get_to(p.diagnosedChance);
    io::from_json(j, dynamic_cast<ConfigRandom::Switch&>(p));
}