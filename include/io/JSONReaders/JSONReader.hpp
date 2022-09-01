#pragma once
#include "nlohmann/json.hpp"
#include <filesystem>
#include <string_view>
#include <vector>

namespace io {
class JSONReader {
    nlohmann::json fileData;

public:
    JSONReader(std::filesystem::path p);

    const nlohmann::json& getData() const;
    const nlohmann::json& getSubData(std::string_view key);
    const nlohmann::json& getSubData(std::string key);
    /*
    T type requires to have a function like this, or be a built in easy std type
    void from_json(const json& j, person& p) {
        j.at("name").get_to(p.name);
        j.at("address").get_to(p.address);
        j.at("age").get_to(p.age);
    }
    */
    template<typename T>
    std::vector<T> parseToArrayofObject() const;
};
}
