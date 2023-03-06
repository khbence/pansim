#pragma once
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

namespace io {
struct Agents {
    struct Person {
        struct Location {
            unsigned typeID;
            std::string locID;
        };

        unsigned char age;
        std::string sex, preCond, state;
        unsigned typeID;
        std::vector<Location> locations;
        bool diagnosed = false;
    };

    std::vector<Person> people;
};

void from_json(const nlohmann::json& j, Agents& p);
void from_json(const nlohmann::json& j, Agents::Person& p);
void from_json(const nlohmann::json& j, Agents::Person::Location& p);
}// namespace io