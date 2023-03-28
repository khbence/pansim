#pragma once
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include "fmt/core.h"
#include <typeinfo>

namespace io {
class JSONReader {
public:
    template<typename JSONParsableType>
    static JSONParsableType parseFile(std::filesystem::path path) {
        fmt::print("Parsing file {} into {} struct", path.string(), typeid(JSONParsableType).name());
        std::ifstream f(path);
        return nlohmann::json::parse(f).get<JSONParsableType>();
    }
};
}
