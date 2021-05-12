#include "smallTools.h"

std::string separator() {
#ifdef _WIN32
    return std::string("\\");
#else
    return std::string("/");
#endif
}
