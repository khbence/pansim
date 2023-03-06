#pragma once
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

namespace io {
    struct AgentTypes {
        struct Type {
            struct Schedule {
                int type;
                double chance, start, end, duration;
            };

            std::string name;
            unsigned ID;
            std::vector<Schedule> schedule;
        };

        std::vector<Type> types;
    };

    void from_json(const nlohmann::json& j, AgentTypes& p);
    void from_json(const nlohmann::json& j, AgentTypes::Type& p);
    void from_json(const nlohmann::json& j, AgentTypes::Type::Schedule& p);
} // namespace io