#include "smallTools.h"

namespace smallTools {
std::filesystem::path createRootDirs(std::string_view rawPath) {
    auto path = std::filesystem::path{rawPath};
    std::filesystem::create_directories(path.remove_filename());
    return std::filesystem::path{rawPath};
}
}// namespace smallTools
