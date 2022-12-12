#pragma once
#include <filesystem>
#include <string_view>

namespace smallTools {
std::filesystem::path createRootDirs(std::string_view rawPath);
}// namespace smallTools
