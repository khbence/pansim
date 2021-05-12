#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct Locations : public jsond::JSONDecodable<Locations> {
        struct Place : public jsond::JSONDecodable<Place> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, ID);
            DECODABLE_MEMBER(int, type);
            DECODABLE_MEMBER(int, essential);
            DECODABLE_MEMBER(std::vector<double>, coordinates);
            DECODABLE_MEMBER(double, infectious);
            DECODABLE_MEMBER(int, area);
            DECODABLE_MEMBER(std::string, state);
            DECODABLE_MEMBER(int, capacity);
            DECODABLE_MEMBER(std::vector<int>, ageInter);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(std::vector<Place>, places);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser