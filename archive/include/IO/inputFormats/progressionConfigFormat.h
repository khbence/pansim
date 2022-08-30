#pragma once
#include "JSONDecoder.h"
#include <vector>
#include "string"

namespace parser {
    struct ProgressionDirectory : public jsond::JSONDecodable<ProgressionDirectory> {
        struct StateInformation : public jsond::JSONDecodable<StateInformation> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::vector<std::string>, stateNames);
            DECODABLE_MEMBER(std::string, firstInfectedState);
            DECODABLE_MEMBER(std::string, nonCOVIDDeadState);
            DECODABLE_MEMBER(std::vector<std::string>, susceptibleStates);
            DECODABLE_MEMBER(std::vector<std::string>, infectedStates);
            END_MEMBER_DECLARATIONS();
        };

        struct ProgressionFile : public jsond::JSONDecodable<ProgressionFile> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, fileName);
            DECODABLE_MEMBER(std::vector<int>, age);
            DECODABLE_MEMBER(std::string, preCond);
            END_MEMBER_DECLARATIONS();
        };

        struct SingleState : public jsond::JSONDecodable<SingleState> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, stateName);
            DECODABLE_MEMBER(std::string, WB);
            DECODABLE_MEMBER(float, infectious);
            DECODABLE_MEMBER(float, accuracyPCR);
            DECODABLE_MEMBER(float, accuracyAntigen);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(StateInformation, stateInformation);
        DECODABLE_MEMBER(std::vector<ProgressionFile>, transitionMatrices);
        DECODABLE_MEMBER(std::vector<SingleState>, states);
        END_MEMBER_DECLARATIONS();
    };

}// namespace parser