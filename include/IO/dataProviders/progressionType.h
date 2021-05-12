#pragma once
#include <string>
#include <utility>
#include "progressionConfigFormat.h"

struct ProgressionType {
    unsigned ageBegin;
    unsigned ageEnd;
    std::string preCond;

    ProgressionType(const parser::ProgressionDirectory::ProgressionFile& file);
    friend bool operator<(const ProgressionType& lhs, const ProgressionType& rhs);
    friend bool operator<(const ProgressionType& lhs, const std::pair<unsigned, std::string>& rhs);
    friend bool operator<(const std::pair<unsigned, std::string>& lhs, const ProgressionType& rhs);
};