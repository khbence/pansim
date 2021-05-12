#pragma once
#include <vector>

#include "JSONDecoder.h"

namespace parser {
    struct TransitionFormat : public jsond::JSONDecodable<TransitionFormat> {
        struct SingleState : public jsond::JSONDecodable<SingleState> {
            struct Progression : public jsond::JSONDecodable<Progression> {
                BEGIN_MEMBER_DECLARATIONS();

                DECODABLE_MEMBER(std::string, name);
                DECODABLE_MEMBER(double, chance);
                DECODABLE_MEMBER(bool, isBadProgression);

                END_MEMBER_DECLARATIONS();
            };

            BEGIN_MEMBER_DECLARATIONS();

            DECODABLE_MEMBER(std::string, stateName);
            DECODABLE_MEMBER(int, avgLength);
            DECODABLE_MEMBER(int, maxlength);
            DECODABLE_MEMBER(std::vector<Progression>, progressions);

            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(std::vector<::parser::TransitionFormat::SingleState>, states);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser