#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct Parameters : public jsond::JSONDecodable<Parameters> {
        struct Sex : public jsond::JSONDecodable<Sex> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, name);
            DECODABLE_MEMBER(double, symptoms);
            END_MEMBER_DECLARATIONS();
        };

        struct Age : public jsond::JSONDecodable<Age> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(int, from);
            DECODABLE_MEMBER(int, to);
            DECODABLE_MEMBER(double, symptoms);
            DECODABLE_MEMBER(double, transmission);
            END_MEMBER_DECLARATIONS();
        };

        struct PreCondition : public jsond::JSONDecodable<PreCondition> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, ID);
            DECODABLE_MEMBER(std::string, condition);
            DECODABLE_MEMBER(double, symptoms);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(std::vector<Sex>, sex);
        DECODABLE_MEMBER(std::vector<Age>, age);
        DECODABLE_MEMBER(std::vector<PreCondition>, preCondition);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser