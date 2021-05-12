#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct ConfigRandom : public jsond::JSONDecodable<ConfigRandom> {
        struct LocationTypeChance : public jsond::JSONDecodable<LocationTypeChance> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(unsigned, value);
            DECODABLE_MEMBER(double, chance);
            END_MEMBER_DECLARATIONS();
        };

        struct PreCondChance : public jsond::JSONDecodable<PreCondChance> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, value);
            DECODABLE_MEMBER(double, chance);
            END_MEMBER_DECLARATIONS();
        };

        struct StatesForAge : public jsond::JSONDecodable<StatesForAge> {
            struct Distribution : public jsond::JSONDecodable<Distribution> {
                BEGIN_MEMBER_DECLARATIONS();
                DECODABLE_MEMBER(std::string, value);
                DECODABLE_MEMBER(long double, chance);
                DECODABLE_MEMBER(long double, diagnosedChance);
                END_MEMBER_DECLARATIONS();
            };
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(int, ageStart);
            DECODABLE_MEMBER(int, ageEnd);
            DECODABLE_MEMBER(std::vector<Distribution>, distribution);
            END_MEMBER_DECLARATIONS();
        };

        struct AgentTypeChance : public jsond::JSONDecodable<AgentTypeChance> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(unsigned, value);
            DECODABLE_MEMBER(double, chance);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(double, irregulalLocationChance);
        DECODABLE_MEMBER(std::vector<LocationTypeChance>, locationTypeDistibution);
        DECODABLE_MEMBER(std::vector<PreCondChance>, preCondDistibution);
        DECODABLE_MEMBER(std::vector<StatesForAge>, stateDistibution);
        DECODABLE_MEMBER(std::vector<AgentTypeChance>, agentTypeDistribution);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser