#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct ConfigRandom : public jsond::JSONDecodable<ConfigRandom> {
        struct IrregularChances : public jsond::JSONDecodable<IrregularChances> {
            struct Detail : public jsond::JSONDecodable<Detail> {
                struct Switch : public jsond::JSONDecodable<Switch> {
                    BEGIN_MEMBER_DECLARATIONS();
                    DECODABLE_MEMBER(std::string, value);
                    DECODABLE_MEMBER(double, chance);
                    END_MEMBER_DECLARATIONS();
                };

                BEGIN_MEMBER_DECLARATIONS();
                DECODABLE_MEMBER(std::string, value);
                DECODABLE_MEMBER(double, chanceForType);
                DECODABLE_MEMBER(double, chanceFromAllIrregular);
                DECODABLE_MEMBER(std::vector<Switch>, switchedToWhat);
                END_MEMBER_DECLARATIONS();
            };

            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(double, generalChance);
            DECODABLE_MEMBER(std::vector<Detail>, detailsOfChances);
            END_MEMBER_DECLARATIONS();
        };

        struct LocationTypeChance : public jsond::JSONDecodable<LocationTypeChance> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, value);
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
            DECODABLE_MEMBER(std::string, value);
            DECODABLE_MEMBER(double, chance);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(IrregularChances, irregularLocationChance);
        DECODABLE_MEMBER(std::vector<LocationTypeChance>, locationTypeDistribution);
        DECODABLE_MEMBER(std::vector<PreCondChance>, preCondDistribution);
        DECODABLE_MEMBER(std::vector<StatesForAge>, stateDistribution);
        DECODABLE_MEMBER(std::vector<AgentTypeChance>, agentTypeDistribution);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser