#include "progressionType.h"
#include <cassert>

ProgressionType::ProgressionType(const parser::ProgressionDirectory::ProgressionFile& file)
    : ageBegin(file.age[0]), ageEnd(file.age[1]), preCond(file.preCond) {}


bool operator<(const ProgressionType& lhs, const ProgressionType& rhs) {
    if (lhs.ageBegin == rhs.ageBegin) {
        assert(lhs.ageEnd == rhs.ageEnd);
        return lhs.preCond < rhs.preCond;
    }
    return lhs.ageBegin < rhs.ageBegin;
}

bool operator<(const ProgressionType& lhs, const std::pair<unsigned, std::string>& rhs) {
    if (lhs.ageBegin < rhs.first) {
        if (rhs.first <= lhs.ageEnd) { return lhs.preCond < rhs.second; }
        return true;
    }
    return false;
}

bool operator<(const std::pair<unsigned, std::string>& lhs, const ProgressionType& rhs) {
    if (lhs.first < rhs.ageEnd) {
        if (rhs.ageBegin <= lhs.first) { return lhs.second < rhs.preCond; }
        return true;
    }
    return false;
}