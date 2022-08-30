#pragma once
#include "JSONDecoder.h"
#include <string>
#include <vector>

namespace parser {
    struct LocationTypes : public jsond::JSONDecodable<LocationTypes> {
        struct Type : public jsond::JSONDecodable<Type> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(int, ID);
            DECODABLE_MEMBER(std::string, name);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(unsigned, publicSpace);
        DECODABLE_MEMBER(unsigned, home);
        DECODABLE_MEMBER(unsigned, hospital);
        DECODABLE_MEMBER(unsigned, doctor);
        DECODABLE_MEMBER(unsigned, school);
        DECODABLE_MEMBER(unsigned, classroom);
        DECODABLE_MEMBER(unsigned, work);
        DECODABLE_MEMBER(unsigned, nurseryhome);
        DECODABLE_MEMBER(std::vector<Type>, types);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser