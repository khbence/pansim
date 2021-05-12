#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct ClosureRules : public jsond::JSONDecodable<ClosureRules> {
        struct Rule : public jsond::JSONDecodable<Rule> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, name);
            DECODABLE_MEMBER(std::string, conditionType);
            DECODABLE_MEMBER(double, threshold);
            DECODABLE_MEMBER(std::string, parameter);
            DECODABLE_MEMBER(int, closeAfter);
            DECODABLE_MEMBER(int, openAfter);
            DECODABLE_MEMBER(std::vector<int>, locationTypesToClose);
            END_MEMBER_DECLARATIONS();
        };
        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(std::vector<Rule>, rules);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser