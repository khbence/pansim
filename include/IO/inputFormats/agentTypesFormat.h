#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct AgentTypes : public jsond::JSONDecodable<AgentTypes> {
        struct Type : public jsond::JSONDecodable<Type> {
            struct ScheduleUnique : public jsond::JSONDecodable<ScheduleUnique> {
                struct Event : public jsond::JSONDecodable<Event> {
                    BEGIN_MEMBER_DECLARATIONS();
                    DECODABLE_MEMBER(int, type);
                    DECODABLE_MEMBER(double, chance);
                    DECODABLE_MEMBER(double, start);
                    DECODABLE_MEMBER(double, end);
                    DECODABLE_MEMBER(double, duration);
                    END_MEMBER_DECLARATIONS();
                };

                BEGIN_MEMBER_DECLARATIONS();
                DECODABLE_MEMBER(std::string, WB);
                DECODABLE_MEMBER(std::string, dayType);
                DECODABLE_MEMBER(std::vector<Event>, schedule);
                END_MEMBER_DECLARATIONS();
            };

            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, name);
            DECODABLE_MEMBER(int, ID);
            DECODABLE_MEMBER(std::vector<ScheduleUnique>, schedulesUnique);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(std::vector<Type>, types);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser