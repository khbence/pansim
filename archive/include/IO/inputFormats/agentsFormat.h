#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct Agents : public jsond::JSONDecodable<Agents> {
        struct Person : public jsond::JSONDecodable<Person> {
            struct Location : public jsond::JSONDecodable<Location> {
                BEGIN_MEMBER_DECLARATIONS();
                DECODABLE_MEMBER(int, typeID);
                DECODABLE_MEMBER(std::string, locID);
                END_MEMBER_DECLARATIONS();
            };

            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(int, age);
            DECODABLE_MEMBER(std::string, sex);
            DECODABLE_MEMBER(std::string, preCond);
            DECODABLE_MEMBER(std::string, state);
            DECODABLE_MEMBER(int, typeID);
            DECODABLE_MEMBER(std::vector<Location>, locations);
            END_MEMBER_DECLARATIONS();
            bool diagnosed;
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(std::vector<Person>, people);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser