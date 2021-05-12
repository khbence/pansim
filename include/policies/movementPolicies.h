#pragma once
#include <iostream>
#include "timeHandler.h"
#include "datatypes.h"
#include "cxxopts.hpp"
#include "operators.h"
#include "locationTypesFormat.h"

#define MIN(a,b) (a)<(b)?(a):(b)
template<typename SimulationType>
class NoMovement {
public:
    // add program parameters if we need any, this function got called already
    // from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data, unsigned cemeteryID) {}

    void planLocations(Timehandler simTime, unsigned timeStep) {}
    void movement(Timehandler simTime, unsigned timeStep) {}
};

template<typename SimulationType>
class DummyMovement {
protected:
    thrust::device_vector<unsigned> stepsUntilMove;

public:
    // add program parameters if we need any, this function got called already
    // from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data, unsigned cemeteryID) {}

    void planLocations(Timehandler simTime, unsigned timeStep) {
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();

        if (stepsUntilMove.size() == 0) stepsUntilMove.resize(numberOfAgents);
        thrust::fill(stepsUntilMove.begin(), stepsUntilMove.end(), 0u);
    }

    void movement(Timehandler simTime, unsigned timeStep) {
        //PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationAgentList = realThis->locs->locationAgentList;
        thrust::device_vector<unsigned>& locationListOffsets = realThis->locs->locationListOffsets;
        thrust::device_vector<unsigned>& locationIdsOfAgents = realThis->locs->locationIdsOfAgents;
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();
        unsigned numberOfLocations = locationListOffsets.size() - 1;

        thrust::for_each(thrust::make_zip_iterator(
                             thrust::make_tuple(agentLocations.begin(), stepsUntilMove.begin())),
            thrust::make_zip_iterator(
                thrust::make_tuple(agentLocations.end(), stepsUntilMove.end())),
            [numberOfLocations] HD(thrust::tuple<unsigned&, unsigned&> tuple) {
                auto& location = thrust::get<0>(tuple);
                auto& stepsUntilMove = thrust::get<1>(tuple);
                if (stepsUntilMove == 0) {
                    location = RandomGenerator::randomUnsigned(numberOfLocations);
                    stepsUntilMove =
                        RandomGenerator::randomUnsigned(144 / 4);// Move 4 times per day on average
                }
                stepsUntilMove--;
            });
        Util::updatePerLocationAgentLists(
            agentLocations, locationIdsOfAgents, locationAgentList, locationListOffsets);
    }
};

namespace RealMovementOps {
    [[nodiscard]] HD unsigned findActualLocationForType(unsigned agent,
        unsigned locType,
        unsigned long* locationOffsetPtr,
        unsigned* possibleLocationsPtr,
        unsigned* possibleTypesPtr, unsigned homeType, unsigned schoolType, unsigned workType, int retry, bool*locationStatesPtr) {
        if (locType == homeType || locType == schoolType || locType == workType) {
            for (unsigned i = locationOffsetPtr[agent]; i < locationOffsetPtr[agent + 1]; i++) {
                if (locType == possibleTypesPtr[i]) return possibleLocationsPtr[i];
            }
            return std::numeric_limits<unsigned>::max();
        }
        //count number of possible locations
        unsigned counter = 0;
        unsigned pos = 0;
        for (unsigned i = locationOffsetPtr[agent]; i < locationOffsetPtr[agent + 1]; i++) {
            if (locType == possibleTypesPtr[i]) {
                if (counter == 0) pos = i;
                counter++;
            }
        }
        if (counter == 1) return possibleLocationsPtr[pos];
        else if (counter > 1) {
            //Pick one at random
            unsigned counter2 = counter;
            unsigned randIdx = counter2;
            while (true) {
                while (randIdx == counter2) randIdx = RandomGenerator::randomUnsigned(counter);
                counter2 = 0;
                for (unsigned i = pos; i < locationOffsetPtr[agent + 1]; i++) {
                    if (locType == possibleTypesPtr[i]) {
                        if (counter2 == randIdx) {
                            if (retry && locationStatesPtr[possibleLocationsPtr[i]]==false) {retry = 0; break;}
                            else return possibleLocationsPtr[i];
                        }
                        counter2++;
                    }
                }
            }
        }
        return std::numeric_limits<unsigned>::max();

        // printf("locType %d not found for agent %d - locationOffsets:
        // %d-%d\n", locType, agent, locationOffsetPtr[agent],
        // locationOffsetPtr[agent+1]);
        
    }

    template<typename PPState, typename AgentMeta, typename LocationType>
    struct MovementArguments {
        MovementArguments() : simTime(0u) {}
        unsigned* stepsUntilMovePtr;
        PPState* agentStatesPtr;
        AgentMeta* agentMetaDataPtr;
        unsigned* agentTypesPtr;
        bool* diagnosedPtr;
        bool* quarantinedPtr;
        AgentStats* agentStatsPtr;
        unsigned* eventOffsetPtr;
        AgentTypeList::Event* eventsPtr;
        unsigned* agentLocationsPtr;
        unsigned long* locationOffsetPtr;
        unsigned* possibleLocationsPtr;
        unsigned* possibleTypesPtr;
        bool* locationStatesPtr;
        bool* stayedHomePtr;
        unsigned *closedUntilPtr;
        unsigned* locationCapacitiesPtr;
        unsigned* locationQuarantineUntilPtr;
        unsigned quarantinePolicy;
        unsigned quarantineLength;
        Days day;
        unsigned hospitalType;
        unsigned homeType;
        unsigned publicPlaceType;
        unsigned doctorType;
        TimeDay simTime;
        unsigned timeStep;
        unsigned timestamp;
        unsigned tracked;
        unsigned cemeteryLoc;
        unsigned schoolType;
        unsigned classroomType;
        unsigned workType;
        LocationType* locationTypePtr;
        uint8_t *noWorkAgentPtr;
        unsigned curfewBegin;
        unsigned curfewEnd;
        bool enableCurfew;
        unsigned schoolAgeRestriction;

    };

    template<typename PPState, typename AgentMeta, typename LocationType>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
        void
        quarantineAgent(unsigned i, MovementArguments<PPState, AgentMeta, LocationType>& a, unsigned until) {
        if (a.quarantinePolicy == 0) return;
        if (a.agentStatsPtr[i].diagnosedTimestamp > 0 && a.agentStatesPtr[i].isInfected() == false) return;
        a.quarantinedPtr[i] = true;
        a.agentStatsPtr[i].quarantinedTimestamp = a.timestamp;
        unsigned previousQuarantineUntil = a.agentStatsPtr[i].quarantinedUntilTimestamp;
        a.agentStatsPtr[i].quarantinedUntilTimestamp = until;
        a.agentStatsPtr[i].daysInQuarantine += (until-a.timestamp)/(24*60/a.timeStep);

        // If agent was also diagnosed (is sick with COVID)
        if (a.diagnosedPtr[i]) {
            // Place home under quarantine
            unsigned myHome = RealMovementOps::findActualLocationForType(
                i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                a.homeType, a.schoolType, a.workType,0,nullptr);
            if (a.quarantinePolicy > 1) {// Home under quarantine for 2 weeks from now
                a.locationQuarantineUntilPtr[myHome] = until;// TODO: quarantine period
                if (i == a.tracked) printf("\tFlagging home as quarantined: %d\n", myHome);
                // if (myHome==2149) printf("LOCATION 2149 quarantined until %d
                // because agent %d got
                // hospitalized\n",a.locationQuarantineUntilPtr[myHome],i);
            }// Place work/classroom under quarantine
            if (a.quarantinePolicy > 2 && previousQuarantineUntil < a.timestamp) {
                unsigned classroom = RealMovementOps::findActualLocationForType(i,
                    a.classroomType,
                    a.locationOffsetPtr,
                    a.possibleLocationsPtr,
                    a.possibleTypesPtr,
                    a.homeType, a.schoolType, a.workType,0,nullptr);
                unsigned work = RealMovementOps::findActualLocationForType(
                    i, a.workType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                    a.homeType, a.schoolType, a.workType,0,nullptr);
                unsigned toClose[2] = { classroom, work };
                for (unsigned loc : toClose) {
                    if (loc != std::numeric_limits<unsigned>::max()
                        && (a.locationTypePtr[loc] == a.workType 
                            || a.locationTypePtr[loc] == a.classroomType)
                        //only quarantining work & classrooms
                        //&& a.locationTypePtr[loc] != a.doctorType
                        //&& a.locationTypePtr[loc] != a.hospitalType
                        && a.locationQuarantineUntilPtr[loc] < a.timestamp
                        && a.locationStatesPtr[loc] == true
                        && a.closedUntilPtr[loc] < a.timestamp) {
                        if (i == a.tracked)
                            printf("\tFlagging work/classroom as quarantined: %d\n", loc);
                        a.locationQuarantineUntilPtr[loc] = until;// TODO: quarantine period
                    }
                }
            }
        }
    }

        template<typename PPState, typename AgentMeta, typename LocationType>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
        void
        checkLarger(unsigned i, MovementArguments<PPState, AgentMeta, LocationType>& a) {
            /*      if (a.stepsUntilMovePtr[i] >  a.simTime.getStepsUntilMidnight(a.timeStep)) {
                printf("WARN LARGER %d > %d\n", a.stepsUntilMovePtr[i],  a.simTime.getStepsUntilMidnight(a.timeStep));
            }*/
        }

    template<typename PPState, typename AgentMeta, typename LocationType>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
        void
        doMovement(unsigned i, MovementArguments<PPState, AgentMeta, LocationType>& a) {
        unsigned& agentType = a.agentTypesPtr[i];

        //if not dead or not in hospital (covid or non-covid) go home at curfew
        if (a.enableCurfew && a.curfewBegin == a.simTime.getMinutes()/a.timeStep) {
            states::WBStates wBState = a.agentStatesPtr[i].getWBState();
            bool deadOrHospitalized  = (wBState == states::WBStates::D || wBState == states::WBStates::S);
            bool hospitalizedWithNonCOVID = (a.agentStatsPtr[i].hospitalizedTimestamp <= a.timestamp && 
                  a.agentStatsPtr[i].hospitalizedUntilTimestamp > a.timestamp);
            if (!deadOrHospitalized &&
                !hospitalizedWithNonCOVID)
              if (agentType+1 == 7) { //afternoon shift worker
                //if currently at work, do nothing
                unsigned workplace = RealMovementOps::findActualLocationForType(i, a.workType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                            a.homeType, a.schoolType, a.workType,0,nullptr);
                //if not at workplace, move home, but allow movement later
                if (a.agentLocationsPtr[i] != workplace)
                    a.agentLocationsPtr[i] =
                    RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                a.homeType, a.schoolType, a.workType,0,nullptr);
                    
              } else {
                a.stepsUntilMovePtr[i] = a.simTime.getStepsUntilMidnight(a.timeStep);
                a.agentLocationsPtr[i] =
                    RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                a.homeType, a.schoolType, a.workType,0,nullptr);
                if (a.tracked == i)
                    printf("Agent %d day %d at %d:%d WBState %d moved to home %d due to curfew\n",
                    i,
                    (int)a.day,
                    a.simTime.getMinutes() / 60,
                    a.simTime.getMinutes() % 60,
                    (int)wBState,
                    a.agentLocationsPtr[i]);
                return;
              }
        }

        if (a.stepsUntilMovePtr[i] > 0) {
            a.stepsUntilMovePtr[i]--;
            return;
        }

        if (a.agentStatsPtr[i].quarantinedUntilTimestamp <= a.timestamp) {
            a.quarantinedPtr[i] = false;
        }

        states::WBStates wBState = a.agentStatesPtr[i].getWBState();
        if (wBState == states::WBStates::D) {// If dead, do not go anywhere
            a.stepsUntilMovePtr[i] = std::numeric_limits<unsigned>::max();
            a.agentLocationsPtr[i] = a.cemeteryLoc;
            return;
        }

        unsigned agentHome = RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr,
                                 a.possibleLocationsPtr, a.possibleTypesPtr,
                                 a.homeType, a.schoolType, a.workType,0,nullptr);
        
        //if non-COVID hospitalization, go to hospital
        if (a.agentStatsPtr[i].hospitalizedTimestamp <= a.timestamp && 
            a.agentStatsPtr[i].hospitalizedUntilTimestamp > a.timestamp &&
            wBState != states::WBStates::S && wBState != states::WBStates::D) {

            a.stepsUntilMovePtr[i] = MIN(a.agentStatsPtr[i].hospitalizedUntilTimestamp - a.timestamp - 1,
                                         a.simTime.getStepsUntilMidnight(a.timeStep));
            a.agentLocationsPtr[i] =
                RealMovementOps::findActualLocationForType(i, a.hospitalType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                            a.homeType, a.schoolType, a.workType,0,nullptr);
            if (i == a.tracked) {
                printf("Agent %d of type %d day %d at %d:%d WBState %d in hospital %d due to non-COVID hospitalization between %d-%d\n",
                i,
                agentType + 1,
                (int)a.day,
                a.simTime.getMinutes() / 60,
                a.simTime.getMinutes() % 60,
                (int)wBState,
                a.agentLocationsPtr[i],
                a.agentStatsPtr[i].hospitalizedTimestamp,
                a.agentStatsPtr[i].hospitalizedUntilTimestamp
                );
            }
            checkLarger(i,a);
            return;
        }

        if (wBState == states::WBStates::S) {// go to hospital if in serious condition
            a.stepsUntilMovePtr[i] = a.simTime.getStepsUntilMidnight(a.timeStep);
            a.agentLocationsPtr[i] = RealMovementOps::findActualLocationForType(
                i, a.hospitalType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                a.homeType, a.schoolType, a.workType,0,nullptr);
            if (i == a.tracked) {
                printf(
                    "Agent %d of type %d day %d at %d:%d WBState %d in "
                    "hospital %d\n",
                    i,
                    agentType + 1,
                    (int)a.day,
                    a.simTime.getMinutes() / 60,
                    a.simTime.getMinutes() % 60,
                    (int)wBState,
                    a.agentLocationsPtr[i]);
            }
            // If not diagnosed before, diagnose & quarantine
            if (!a.diagnosedPtr[i] && a.agentStatesPtr[i].isInfectious()) {
                a.diagnosedPtr[i] = true;
                a.agentStatsPtr[i].diagnosedTimestamp = a.timestamp;
                if (a.simTime.getStepsUntilMidnight(a.timeStep)
                    == 24 * 60 / a.timeStep)// is it midnight, and agent got S
                                            // due to disease progression?
                    a.agentStatsPtr[i].diagnosedTimestamp++;// shift timestamp by 1 to avoid
                                                            // being counted as random test in
                                                            // TestingPolicy

                RealMovementOps::quarantineAgent(i, a, a.timestamp + a.quarantineLength * 24 * 60 / a.timeStep);
            }
            checkLarger(i,a);
            return;
        }

        // Is agent currently in a place under quarantine
        if (a.quarantinePolicy > 1
            && a.timestamp < a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]]
            && (a.locationTypePtr[a.agentLocationsPtr[i]] == a.homeType
                || a.locationTypePtr[a.agentLocationsPtr[i]]
                       == a.schoolType// Only send agent to quarantine if this
                                      // is home, work or school
                || a.locationTypePtr[a.agentLocationsPtr[i]] == a.classroomType
                || a.locationTypePtr[a.agentLocationsPtr[i]] == a.workType)) {
            if (a.quarantinedPtr[i] == false) {
                if (i == a.tracked)
                    printf(
                        "Agent %d of type %d day %d at %d:%d location %d is "
                        "quarantined, staying at home until %d\n",
                        i,
                        agentType + 1,
                        (int)a.day,
                        a.simTime.getMinutes() / 60,
                        a.simTime.getMinutes() % 60,
                        a.agentLocationsPtr[i],
                        a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]]);
                RealMovementOps::quarantineAgent(
                    i, a, a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]]);
            }
            //if now quarantined
            if (a.quarantinedPtr[i] == true) {
                a.stepsUntilMovePtr[i] =
                    MIN(a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]] - a.timestamp - 1,
                        a.simTime.getStepsUntilMidnight(a.timeStep));

                if (i == a.tracked) {
                    printf(
                        "Agent %d of type %d day %d at %d:%d WBState %d at "
                        "location %d under quarantine (1), quarantined %d-%d "
                        "locationQuarantineUntil %d timestamp %d\n",
                        i,
                        agentType + 1,
                        (int)a.day,
                        a.simTime.getMinutes() / 60,
                        a.simTime.getMinutes() % 60,
                        (int)wBState,
                        a.agentLocationsPtr[i],
                        a.agentStatsPtr[i].quarantinedTimestamp,
                        a.agentStatsPtr[i].quarantinedUntilTimestamp,
                        a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]],
                        a.timestamp);
                }

                // If not home, send home
                unsigned homeLocation = RealMovementOps::findActualLocationForType(
                    i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                    a.homeType, a.schoolType, a.workType,0,nullptr);
                if (homeLocation != a.agentLocationsPtr[i]) {
                    a.agentLocationsPtr[i] = homeLocation;
                    // TODO: quarantine whole home??
                    // unsigned until =
                    // a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]];
                    // a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]] = until;
                }
                checkLarger(i,a);
                if (agentHome != std::numeric_limits<unsigned>::max() 
                        && a.agentLocationsPtr[i] != agentHome 
                        && a.stepsUntilMovePtr[i] > 60 / a.timeStep)
                        a.stayedHomePtr[i] = false;
                return;
            }
        }

        // Should agent still be quarantined due to recent diagnosis
        if ((a.quarantinePolicy > 0
                && (a.diagnosedPtr[i]
                    || (a.agentStatsPtr[i].diagnosedTimestamp > 0
                        && (a.timestamp - a.agentStatsPtr[i].diagnosedTimestamp)
                               < a.quarantineLength * 24 * 60 / a.timeStep)))// stay home if diagnosed or
                                                                // quarantine has not
                                                                // expired
            || (a.quarantinePolicy > 0 && a.quarantinedPtr[i])) {

            // Diagnosed, but not yet quarantined
            if (a.quarantinePolicy > 0 && !a.quarantinedPtr[i]) {
                RealMovementOps::quarantineAgent(i, a, a.timestamp + a.quarantineLength * 24 * 60 / a.timeStep);
                if (i == a.tracked && a.quarantinedPtr[i]) {
                    printf(
                        "Agent %d of type %d day %d at %d:%d WBState %d was "
                        "recently diagnosed, enforcing quarantine: diagnosed "
                        "%d diagnosedTimestamp %d, current timestamp %d\n",
                        i,
                        agentType + 1,
                        (int)a.day,
                        a.simTime.getMinutes() / 60,
                        a.simTime.getMinutes() % 60,
                        (int)wBState,
                        a.diagnosedPtr[i],
                        a.agentStatsPtr[i].diagnosedTimestamp,
                        a.timestamp);
                }
            }

            if (a.quarantinedPtr[i]) {
                // Stay in quarantine at home
                a.agentLocationsPtr[i] = RealMovementOps::findActualLocationForType(
                    i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                    a.homeType, a.schoolType, a.workType,0,nullptr);

                // if less than 2 weeks since diagnosis/quarantine, stay where agent
                // already is
                a.stepsUntilMovePtr[i] = a.simTime.getStepsUntilMidnight(a.timeStep);
                if (i == a.tracked) {
                    printf(
                        "Agent %d of type %d day %d at %d:%d WBState %d still "
                        "quarantined (2): diagnosed %d diagnosedTimestamp %d, "
                        "personal quarantine until %d, current timestamp %d\n",
                        i,
                        agentType + 1,
                        (int)a.day,
                        a.simTime.getMinutes() / 60,
                        a.simTime.getMinutes() % 60,
                        (int)wBState,
                        a.diagnosedPtr[i],
                        a.agentStatsPtr[i].diagnosedTimestamp,
                        a.agentStatsPtr[i].quarantinedUntilTimestamp,
                        a.timestamp);
                }
                checkLarger(i,a);
                if (agentHome != std::numeric_limits<unsigned>::max() 
                        && a.agentLocationsPtr[i] != agentHome 
                        && a.stepsUntilMovePtr[i] > 60 / a.timeStep)
                        a.stayedHomePtr[i] = false;
                return;
            }
        }

        unsigned agentTypeOffset = AgentTypeList::getOffsetIndex(agentType, wBState, a.day);
        unsigned eventsBegin = a.eventOffsetPtr[agentTypeOffset];
        unsigned eventsEnd = a.eventOffsetPtr[agentTypeOffset + 1];

        int activeEventsBegin = -1;
        int activeEventsEnd = -1;

        // Here we assume if multiple events are given for the same timeslot,
        // they all start & end at the same time
        for (unsigned j = eventsBegin; j < eventsEnd; j++) {
            if (a.simTime >= a.eventsPtr[j].start && a.simTime < a.eventsPtr[j].end
                && activeEventsBegin == -1)
                activeEventsBegin = j;
            if (a.simTime < a.eventsPtr[j].start) {
                activeEventsEnd = j;
                break;
            }
            if (j == eventsEnd-1) {
                if (a.simTime >= a.eventsPtr[j].start && a.simTime < a.eventsPtr[j].end)
                    activeEventsEnd = eventsEnd;
            }
        }
        if (i == a.tracked)
            printf(
                "Agent %d of type %d day %d at %d:%d WBState %d activeEvents: "
                "%d-%d\n",
                i,
                agentType + 1,
                (int)a.day,
                a.simTime.getMinutes() / 60,
                a.simTime.getMinutes() % 60,
                (int)wBState,
                activeEventsBegin,
                activeEventsEnd);

        // Possibilities:
        // 1 both are -1 -> no more events for that day. Should be home if
        // wBState != S, or at hospital if S 
        // 2 Begin != -1, End == -1 -> last event for the day. Move there (if needed pick randomly) 
        // 3 Begin == -1, End != -1 -> no events right now, but there will be some later
        //      3a if less than 30 mins until next possible event, then stay
        //      here 3b if 30-60 to next possible event, should go to public
        //      place (type 0) 3c more than 60 mins, then go home
        // 4 neither -1, then pick randomly between one of the events

        // ISSUES:
        // do we forcibly finish at midnight?? What if the duration goes beyond
        // that?
        unsigned newLocationType = std::numeric_limits<unsigned>::max();

        // Case 1
        if (activeEventsBegin == -1 && activeEventsEnd == -1) {
            newLocationType = wBState == states::WBStates::S
                                  ? a.hospitalType
                                  : a.homeType;// Hostpital if sick, home otherwise
            unsigned myHome = RealMovementOps::findActualLocationForType(i,
                newLocationType,
                a.locationOffsetPtr,
                a.possibleLocationsPtr,
                a.possibleTypesPtr,
                a.homeType, a.schoolType, a.workType,0,nullptr);
            a.agentLocationsPtr[i] = myHome;
            a.stepsUntilMovePtr[i] = a.simTime.getStepsUntilMidnight(a.timeStep);
            checkLarger(i,a);
            if (i == a.tracked)
                printf(
                    "\tCase 1- moving to locType %d location %d until midnight "
                    "(for %d steps)\n",
                    newLocationType,
                    myHome,
                    a.stepsUntilMovePtr[i] - 1);
        }
        // Case 2 and 4
        if (activeEventsBegin != -1) {
            unsigned numPotentialEvents = (activeEventsEnd==-1?activeEventsBegin+1:activeEventsEnd) - activeEventsBegin;
            unsigned pickedEventIdx = 0;
            TimeDayDuration basicDuration(0.0);
            if (numPotentialEvents == 1) {
                newLocationType = a.eventsPtr[activeEventsBegin].locationType;
                basicDuration = a.eventsPtr[activeEventsBegin].duration;
            } else {
                double rand = RandomGenerator::randomReal(1.0);
                double threshhold = a.eventsPtr[activeEventsBegin].chance;
                unsigned idx = 0;
                while (rand > threshhold && idx < numPotentialEvents) {
                    idx++;
                    threshhold += a.eventsPtr[activeEventsBegin + idx].chance;
                }
                if (idx==numPotentialEvents) {
                    /*printf("Error, overrun1: %g, agentType %d WB %d, day %d, time %d:%d\n",rand, agentType + 1, wBState,
                (int)a.day,
                a.simTime.getMinutes() / 60,
                a.simTime.getMinutes() % 60);*/
                    idx--;
                }
                pickedEventIdx = idx;
                newLocationType = a.eventsPtr[activeEventsBegin + idx].locationType;
                basicDuration = a.eventsPtr[activeEventsBegin + idx].duration;
            }

            //if agent has to stay home with children, then check to see if is work, and set it to home
            if (a.noWorkAgentPtr[i]!=0 && newLocationType == a.workType) {
                newLocationType = a.homeType;
                numPotentialEvents = 1;
                if (a.quarantinedPtr[i] == false) a.agentStatsPtr[i].daysInQuarantine++;
                if (i == a.tracked) printf("Agent %d not going to work because child at home\n", i);
            }

            unsigned newLocation = RealMovementOps::findActualLocationForType(i,
                newLocationType,
                a.locationOffsetPtr,
                a.possibleLocationsPtr,
                a.possibleTypesPtr,
                a.homeType, a.schoolType, a.workType, numPotentialEvents==1, a.locationStatesPtr);
            // Check if location is open/closed. If closed, go home instead
            unsigned wasClosed = std::numeric_limits<unsigned>::max();
            bool schoolAndTooOld = (newLocationType == a.schoolType || newLocationType == a.classroomType) && a.agentMetaDataPtr[i].getAge() >= a.schoolAgeRestriction;
            if (schoolAndTooOld || ((a.locationStatesPtr[newLocation] == false || a.closedUntilPtr[newLocation]>a.timestamp) && newLocationType != a.workType)) {
                //If closed, but there is another option to go to different type location, try that
                if (numPotentialEvents>1) {
                    double rand = RandomGenerator::randomReal(1.0);
                    double threshhold = (pickedEventIdx == 0) ? 0.0 : a.eventsPtr[activeEventsBegin].chance/(1.0-a.eventsPtr[activeEventsBegin+pickedEventIdx].chance);
                    unsigned idx = 0;
                    while (rand > threshhold && idx < numPotentialEvents) {
                        idx++;
                        threshhold += (pickedEventIdx == idx) ? 0.0 : a.eventsPtr[activeEventsBegin + idx].chance/(1.0-a.eventsPtr[activeEventsBegin+pickedEventIdx].chance);
                    }
                    if (idx==numPotentialEvents) {
                        idx--;
                    }
                    newLocationType = a.eventsPtr[activeEventsBegin + idx].locationType;
                    basicDuration = a.eventsPtr[activeEventsBegin + idx].duration;
                    newLocation = RealMovementOps::findActualLocationForType(i,
                        newLocationType,
                        a.locationOffsetPtr,
                        a.possibleLocationsPtr,
                        a.possibleTypesPtr,
                        a.homeType, a.schoolType, a.workType, 0, nullptr);
                    wasClosed = std::numeric_limits<unsigned>::max();
                    bool schoolAndTooOld2 = (newLocationType == a.schoolType || newLocationType == a.classroomType) && a.agentMetaDataPtr[i].getAge() >= a.schoolAgeRestriction;
                    //is that closed too?
                    if (schoolAndTooOld || (a.locationStatesPtr[newLocation] == false || a.closedUntilPtr[newLocation]>a.timestamp) && newLocationType != a.workType) {
                        wasClosed = newLocation;
                        newLocation = RealMovementOps::findActualLocationForType(
                            i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                            a.homeType, a.schoolType, a.workType, 0, nullptr);
                    }
                } else {
                    wasClosed = newLocation;
                    newLocation = RealMovementOps::findActualLocationForType(
                        i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                        a.homeType, a.schoolType, a.workType, 0, nullptr);
                }
            } else if (newLocationType == a.schoolType) {
                //if classroom closed, don't go to school either
                unsigned myClassroom = RealMovementOps::findActualLocationForType(
                    i, a.classroomType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                    a.homeType, a.schoolType, a.workType, 0, nullptr);
                if (myClassroom != std::numeric_limits<unsigned>::max() &&
                    a.closedUntilPtr[myClassroom] > a.timestamp) {
                        wasClosed = newLocation;
                        newLocation = RealMovementOps::findActualLocationForType(
                            i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                            a.homeType, a.schoolType, a.workType, 0, nullptr); 
                    }
            }
            
            a.agentLocationsPtr[i] = newLocation;
            if (basicDuration.getHours() > 24) {
                a.stepsUntilMovePtr[i] = a.simTime.getStepsUntilMidnight(a.timeStep);
            } else if (activeEventsEnd == -1) {
                if ((a.simTime + basicDuration).isOverMidnight()) {
                    a.stepsUntilMovePtr[i] = a.simTime.getStepsUntilMidnight(a.timeStep);
                } else {
                    // does not last till midnight, but no events afterwards -
                    // spend full duration there
                    a.stepsUntilMovePtr[i] = basicDuration.steps(a.timeStep);
                }
            } else {
                // If duration is less then the beginning of the next move
                // window, then spend full duration here
                if (a.simTime + basicDuration < a.eventsPtr[activeEventsEnd].start) {
                    a.stepsUntilMovePtr[i] = basicDuration.steps(a.timeStep);
                    checkLarger(i,a);
                } else if (a.simTime + basicDuration > a.eventsPtr[activeEventsEnd].end) {
                    a.stepsUntilMovePtr[i] = (a.eventsPtr[activeEventsEnd].end - a.simTime).steps(a.timeStep) - 1;
                    checkLarger(i,a);
                } else {
                    // Otherwise I need to move again randomly between the end
                    // of this duration and the end of next movement window
                    TimeDayDuration window =
                        a.eventsPtr[activeEventsEnd].end - (a.simTime + basicDuration);
                    unsigned st = window.steps(a.timeStep);
                    unsigned randExtra = RandomGenerator::randomUnsigned(st);
                    a.stepsUntilMovePtr[i] = basicDuration.steps(a.timeStep) + randExtra;
                    checkLarger(i,a);
                }
            }
            if (agentType+1 == 7 && a.enableCurfew && (a.curfewBegin <= a.simTime.getMinutes()/a.timeStep || a.curfewEnd > a.simTime.getMinutes()/a.timeStep)) {
                bool workButClosed = (newLocationType == a.workType && (a.locationStatesPtr[newLocation] == false || a.closedUntilPtr[newLocation]>a.timestamp));
                if (workButClosed || (newLocationType != a.workType && newLocationType != a.homeType)) {
                    a.agentLocationsPtr[i] = agentHome;
                    if (i == a.tracked) {
                        printf(
                        "\tCase 2&4- Night shift worker tried moving to locType %d location %d, "
                        "but it's curfew, moving home to %d for %d steps\n",
                        newLocationType,
                        newLocation,
                        agentHome,
                        a.stepsUntilMovePtr[i] - 1);
                    }
                }
            }

            if (i == a.tracked) {
                if (wasClosed == std::numeric_limits<unsigned>::max())
                    printf(
                        "\tCase 2&4- moving to locType %d location %d for %d "
                        "steps\n",
                        newLocationType,
                        newLocation,
                        a.stepsUntilMovePtr[i] - 1);
                else
                    printf(
                        "\tCase 2&4- tried moving to locType %d location %d, "
                        "but was closed, moving home to %d for %d steps\n",
                        newLocationType,
                        wasClosed,
                        newLocation,
                        a.stepsUntilMovePtr[i] - 1);
            }
        }

        // Case 3
        if (activeEventsBegin == -1 && activeEventsEnd != -1) {
            // Randomly decide when the move will occur in the next window:
            TimeDayDuration length =
                a.eventsPtr[activeEventsEnd].end - a.eventsPtr[activeEventsEnd].start;
            unsigned length_steps = length.steps(a.timeStep);
            unsigned randDelay = RandomGenerator::randomUnsigned(length_steps);
            a.stepsUntilMovePtr[i] =
                (a.eventsPtr[activeEventsEnd].start - a.simTime).steps(a.timeStep) + randDelay;
            unsigned timeLeft = a.stepsUntilMovePtr[i];
            // Case 3.a -- less than 30 mins -> stay here
            if (timeLeft < TimeDayDuration(0.3).steps(a.timeStep)) {
                if (i == a.tracked)
                    printf(
                        "\tCase 3a- staying in place for %d steps\n", a.stepsUntilMovePtr[i] - 1);
                // Do nothing - location stays the same
            } else if (timeLeft < TimeDayDuration(1.0).steps(a.timeStep)) {
                newLocationType = a.publicPlaceType;
                unsigned myPublicPlace = RealMovementOps::findActualLocationForType(i,
                    a.publicPlaceType,
                    a.locationOffsetPtr,
                    a.possibleLocationsPtr,
                    a.possibleTypesPtr,
                    a.homeType, a.schoolType, a.workType, 0, nullptr);
                a.agentLocationsPtr[i] = myPublicPlace;
                if (i == a.tracked)
                    printf(
                        "\tCase 3b- moving to public Place type 1 location %d "
                        "for %d steps\n",
                        myPublicPlace,
                        a.stepsUntilMovePtr[i] - 1);
            } else {
                newLocationType = a.homeType;
                unsigned myHome = RealMovementOps::findActualLocationForType(
                    i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                    a.homeType, a.schoolType, a.workType, 0, nullptr);
                a.agentLocationsPtr[i] = myHome;
                if (i == a.tracked)
                    printf(
                        "\tCase 3c- moving to home type 2 location %d for %d "
                        "steps\n",
                        myHome,
                        a.stepsUntilMovePtr[i] - 1);
            }
        }

        // Has agent just gone someplace currently under quarantine
        if (a.quarantinePolicy > 1
            && a.timestamp < a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]]
            && (a.locationTypePtr[a.agentLocationsPtr[i]] == a.homeType
                || a.locationTypePtr[a.agentLocationsPtr[i]]
                       == a.schoolType// Only send agent to quarantine if this
                                      // is home, work or school
                || a.locationTypePtr[a.agentLocationsPtr[i]] == a.classroomType
                || a.locationTypePtr[a.agentLocationsPtr[i]] == a.workType)) {
            //if not currently under quarantine
            if (!a.quarantinedPtr[i]) {
                    RealMovementOps::quarantineAgent(i, a,
                    a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]]);

                if (i == a.tracked && a.quarantinedPtr[i])
                    printf(
                        "Agent %d of type %d day %d at %d:%d location %d is "
                        "quarantined, staying at home until %d\n",
                        i,
                        agentType + 1,
                        (int)a.day,
                        a.simTime.getMinutes() / 60,
                        a.simTime.getMinutes() % 60,
                        a.agentLocationsPtr[i],
                        a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]]);
            }
            if (a.quarantinedPtr[i]) {
                a.stepsUntilMovePtr[i] =
                    MIN(a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]] - a.timestamp - 1,
                        a.simTime.getStepsUntilMidnight(a.timeStep));
                checkLarger(i,a);

                if (i == a.tracked) {
                    printf(
                        "Agent %d of type %d day %d at %d:%d WBState %d at "
                        "location %d under quarantine (1), quarantined %d-%d "
                        "locationQuarantineUntil %d timestamp %d\n",
                        i,
                        agentType + 1,
                        (int)a.day,
                        a.simTime.getMinutes() / 60,
                        a.simTime.getMinutes() % 60,
                        (int)wBState,
                        a.agentLocationsPtr[i],
                        a.agentStatsPtr[i].quarantinedTimestamp,
                        a.agentStatsPtr[i].quarantinedUntilTimestamp,
                        a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]],
                        a.timestamp);
                }

                // If not home, send home
                unsigned homeLocation = RealMovementOps::findActualLocationForType(
                    i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                    a.homeType, a.schoolType, a.workType, 0, nullptr);
                if (homeLocation != a.agentLocationsPtr[i]) {
                    // unsigned until =
                    // a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]];
                    a.agentLocationsPtr[i] = homeLocation;
                    // TODO: quarantine whole home??
                    // a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]] = until;
                }
                checkLarger(i,a);
                if (agentHome != std::numeric_limits<unsigned>::max() 
                        && a.agentLocationsPtr[i] != agentHome 
                        && a.stepsUntilMovePtr[i] > 60 / a.timeStep)
                        a.stayedHomePtr[i] = false;
                return;
            }
        }

        // if (a.locationTypePtr[a.agentLocationsPtr[i]] == a.workType) {
        //     printf("Agent %d  of type %d moved to work type location %d at day %d %d:%d\n",
        //             i, agentType+1, a.agentLocationsPtr[i],
        //             (int)a.day,
        //             a.simTime.getMinutes() / 60,
        //             a.simTime.getMinutes() % 60);
        // }76030
        // if (a.agentLocationsPtr[i] == 76030) {
        //      printf("Agent %d  of type %d moved to location %d type %d at day %d %d:%d\n",
        //             i, agentType+1, a.agentLocationsPtr[i],
        //             a.locationTypePtr[a.agentLocationsPtr[i]],
        //             (int)a.day,
        //             a.simTime.getMinutes() / 60,
        //             a.simTime.getMinutes() % 60);
        // }


        // Diagnosis-related operations
        if (newLocationType == a.hospitalType || newLocationType == a.doctorType) {
            // If previously undiagnosed and
            if (!a.diagnosedPtr[i] && a.agentStatesPtr[i].isInfectious()) {
                a.diagnosedPtr[i] = true;
                a.agentStatsPtr[i].diagnosedTimestamp = a.timestamp;
                if (a.simTime.getStepsUntilMidnight(a.timeStep)
                    == 24 * 60 / a.timeStep)// is it midnight, and agent got S
                                            // due to disease progression?
                    a.agentStatsPtr[i].diagnosedTimestamp++;// shift timestamp by 1 to avoid
                                                            // being counted as random test in
                                                            // TestingPolicy

                if (i == a.tracked) printf("\tDiagnosed at location %d\n", a.agentLocationsPtr[i]);

                RealMovementOps::quarantineAgent(i,
                    a,
                    a.timestamp + a.quarantineLength * 24 * 60 / a.timeStep);// TODO: quarantine period

                // We are not moving the agent - stay here for full duration,
                // potentially infect others when moving next, he will go into
                // quarantine anyway (if enabled)
            }
        }
        checkLarger(i,a);
        if (agentHome != std::numeric_limits<unsigned>::max() 
                        && a.agentLocationsPtr[i] != agentHome 
                        && a.stepsUntilMovePtr[i] > 60 / a.timeStep) {
                        a.stayedHomePtr[i] = false;
                        if (i == a.tracked) printf("\tdid not stay home %d %d\n",a.agentLocationsPtr[i], agentHome);
        }
        a.stepsUntilMovePtr[i]--;
    }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    template<typename PPState, typename AgentMeta, typename LocationType>
    __global__ void doMovementDriver(unsigned numberOfAgents,
        MovementArguments<PPState, AgentMeta, LocationType> a) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numberOfAgents) { RealMovementOps::doMovement(i, a); }
    }
#endif

    template<typename AgentMeta>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
        void
        checkUnderageAtHome(unsigned i, unsigned *noWorkPtr, AgentMeta *agentMetaDataPtr, bool *quarantinedPtr, bool *locationStatesPtr, unsigned *closedUntilPtr, unsigned long *locationOffsetPtr, 
            unsigned *possibleLocationsPtr, unsigned *possibleTypesPtr, unsigned home, unsigned school, unsigned classroom, unsigned timestamp, unsigned schoolAgeRestriction) {
                if (agentMetaDataPtr[i].getAge() > 14) return; //Only underage
                if (quarantinedPtr[i]) {
                    //If quarantined
                    unsigned homeLocation = RealMovementOps::findActualLocationForType(
                            i, home, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr,
                            home, school, classroom, 0, nullptr);
                    if (homeLocation != std::numeric_limits<unsigned>::max())
                        noWorkPtr[homeLocation] = 1;
                } else {
                    //Check if school open/closed
                    unsigned schoolLocation = RealMovementOps::findActualLocationForType(
                    i, school, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr,
                    home, school, classroom, 0, nullptr);
                    unsigned classroomLocation = RealMovementOps::findActualLocationForType(
                    i, classroom, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr,
                    home, school, classroom, 0, nullptr);
                    bool schoolAndTooOld = agentMetaDataPtr[i].getAge() >= schoolAgeRestriction;
                    if (schoolAndTooOld ||
                        (schoolLocation != std::numeric_limits<unsigned>::max() &&
                        (locationStatesPtr[schoolLocation]==false || closedUntilPtr[schoolLocation]>timestamp))
                        || (classroomLocation != std::numeric_limits<unsigned>::max() &&
                        (locationStatesPtr[classroomLocation]==false || closedUntilPtr[classroomLocation]>timestamp))) { //School closed
                        unsigned homeLocation = RealMovementOps::findActualLocationForType(
                                i, home, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr,
                                home, school, classroom, 0, nullptr);
                        if (homeLocation != std::numeric_limits<unsigned>::max())
                            noWorkPtr[homeLocation] = 1;
                        }
                }
        }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    template<typename AgentMeta>
    __global__ void checkUnderageAtHomeDriver(unsigned numberOfAgents,
        unsigned *noWorkPtr, AgentMeta *agentMetaDataPtr, bool *quarantinedPtr, bool *locationStatesPtr, unsigned *closedUntilPtr, unsigned long *locationOffsetPtr, 
            unsigned *possibleLocationsPtr, unsigned *possibleTypesPtr, unsigned home, unsigned school, unsigned classroom, unsigned timestamp, unsigned schoolAgeRestriction) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numberOfAgents) { RealMovementOps::checkUnderageAtHome(i, noWorkPtr, agentMetaDataPtr, quarantinedPtr, locationStatesPtr, closedUntilPtr,
                                    locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr, home, school, classroom, timestamp, schoolAgeRestriction); }
    }
#endif

    template<typename AgentMeta>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
        void
        setNoWorkToday(unsigned i, unsigned *noWorkLocPtr, uint8_t *noWorkAgentPtr, AgentMeta *agentMetaDataPtr, unsigned long *locationOffsetPtr, 
            unsigned *possibleLocationsPtr, unsigned *possibleTypesPtr, unsigned home) {
                if (agentMetaDataPtr[i].getAge() > 26 && agentMetaDataPtr[i].getAge() < 65) {
                    unsigned homeLocation = RealMovementOps::findActualLocationForType(
                                i, home, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr,
                                home, home, home, 0, nullptr);
                        if (homeLocation != std::numeric_limits<unsigned>::max())
                            if (noWorkLocPtr[homeLocation] == 1) { //TODO this is not exactly thread safe on the CPU....
                            #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
                                if (atomicAdd(&noWorkLocPtr[homeLocation],1)==1)
                            #else
                                noWorkLocPtr[homeLocation] = 2;
                            #endif
                                noWorkAgentPtr[i] = 1;
                            }
                }
        }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    template<typename AgentMeta>
    __global__ void setNoWorkTodayDriver(unsigned numberOfAgents,
        unsigned *noWorkLocPtr, uint8_t *noWorkAgentPtr, AgentMeta *agentMetaDataPtr, unsigned long *locationOffsetPtr, 
            unsigned *possibleLocationsPtr, unsigned *possibleTypesPtr, unsigned home) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numberOfAgents) { RealMovementOps::setNoWorkToday(i, noWorkLocPtr, noWorkAgentPtr, agentMetaDataPtr, 
                                    locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr, home); }
    }
#endif
    template<typename PPValues>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
        void
        checkSchoolWorkQuarantine(unsigned i, AgentStats *agentStatsPtr, PPValues *agentStatesPtr, bool *quarantinedPtr, unsigned *locationQuarantineUntilPtr, unsigned long *locationOffsetPtr, 
            unsigned *possibleLocationsPtr, unsigned *possibleTypesPtr, unsigned home, unsigned work, unsigned school, unsigned classroom, unsigned timestamp, unsigned timeStep, unsigned tracked) {
            //If already quarantined, do nothing
            if (quarantinedPtr[i]) return;
            //If immune, do nothing
            if (agentStatsPtr[i].diagnosedTimestamp > 0 && agentStatesPtr[i].isInfected() == false) return;
            //Get work/school/classroom ID
            unsigned workLocation = RealMovementOps::findActualLocationForType(
            i, work, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr,
            home, school, classroom, 0, nullptr);
            unsigned schoolLocation = RealMovementOps::findActualLocationForType(
            i, school, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr,
            home, school, classroom, 0, nullptr);
            unsigned classroomLocation = RealMovementOps::findActualLocationForType(
            i, classroom, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr,
            home, school, classroom, 0, nullptr);

            //Check if school/classroom/work quarantined
            unsigned until = 0;
            if (schoolLocation != std::numeric_limits<unsigned>::max() &&
                locationQuarantineUntilPtr[schoolLocation] > timestamp)
                until = locationQuarantineUntilPtr[schoolLocation];
            else if (classroomLocation != std::numeric_limits<unsigned>::max() &&
                locationQuarantineUntilPtr[classroomLocation] > timestamp) 
                until = locationQuarantineUntilPtr[classroomLocation];
            else if (workLocation != std::numeric_limits<unsigned>::max() &&
                locationQuarantineUntilPtr[workLocation] > timestamp)
                until = locationQuarantineUntilPtr[workLocation];

            //If so, quarantine agent
            if (until > 0) {
                quarantinedPtr[i] = true;
                agentStatsPtr[i].quarantinedTimestamp = timestamp;
                agentStatsPtr[i].quarantinedUntilTimestamp = until;
                agentStatsPtr[i].daysInQuarantine += (until-timestamp)/(24*60/timeStep);
                if (i == tracked) {
                    printf(
                        "Agent %d at %d: "
                        "school/class/work under quarantine, going to quarantine until %d ",
                        i,
                        timestamp,
                        agentStatsPtr[i].quarantinedUntilTimestamp);
                }
            }
        }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    template<typename PPValues>
    __global__ void checkSchoolWorkQuarantineDriver(unsigned numberOfAgents,
        AgentStats *agentStatsPtr, PPValues *agentStatesPtr, bool *quarantinedPtr, unsigned *locationQuarantineUntilPtr, unsigned long *locationOffsetPtr, unsigned *possibleLocationsPtr,
            unsigned *possibleTypesPtr, unsigned home, unsigned work, unsigned school, unsigned classroom, unsigned timestamp, unsigned timeStep, unsigned tracked) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numberOfAgents) { RealMovementOps::checkSchoolWorkQuarantine(i, agentStatsPtr, agentStatesPtr, quarantinedPtr, locationQuarantineUntilPtr, 
                                    locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr, home, work, school, classroom, timestamp, timeStep, tracked); }
    }
#endif



#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
        void
        checkSchoolQuarantine(unsigned i, unsigned *schoolsPtr, unsigned *classroomsPtr, unsigned *classroomOffsetsPtr, unsigned *locationQuarantineUntilPtr, unsigned timestamp) {
            unsigned schoolIdx = schoolsPtr[i];
            if (locationQuarantineUntilPtr[schoolIdx] > timestamp) return;
            unsigned counter = 0;
            unsigned max = 0;
            for (unsigned classOffset = classroomOffsetsPtr[i]; classOffset < classroomOffsetsPtr[i+1]; classOffset++) {
                unsigned classIdx = classroomsPtr[classOffset];
                if (locationQuarantineUntilPtr[classIdx] > timestamp) { 
                    counter++;
                    max = max > locationQuarantineUntilPtr[classIdx] ? max : locationQuarantineUntilPtr[classIdx];
                }
            }
            if (counter > 1) {
                //printf("School %d has %d quarantined classes, quarantining entire school until %d\n", schoolIdx, counter, max);
                locationQuarantineUntilPtr[schoolIdx] = max;
                for (unsigned classOffset = classroomOffsetsPtr[i]; classOffset < classroomOffsetsPtr[i+1]; classOffset++) {
                    unsigned classIdx = classroomsPtr[classOffset];
                    locationQuarantineUntilPtr[classIdx] = max;
                }
            }
        }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __global__ void checkSchoolQuarantineDriver(unsigned numSchools, unsigned *schoolsPtr, unsigned *classroomsPtr, unsigned *classroomOffsetsPtr, unsigned *locationQuarantineUntilPtr, unsigned timestamp) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numSchools) { RealMovementOps::checkSchoolQuarantine(i, schoolsPtr, classroomsPtr, classroomOffsetsPtr, locationQuarantineUntilPtr, timestamp); }
    }
#endif
}// namespace RealMovementOps

template<typename SimulationType>
class RealMovement {
    thrust::device_vector<unsigned> stepsUntilMove;
    thrust::device_vector<unsigned> noWorkLoc; //indicating children at home
    thrust::device_vector<uint8_t> noWorkAgent; //indicating agent staying home because children at home
    unsigned publicSpace;
    unsigned home;
    unsigned hospital;
    unsigned cemeteryLoc;
    unsigned doctor;
    unsigned tracked;
    unsigned quarantinePolicy;
    unsigned quarantineLength;
    unsigned school;
    unsigned classroom;
    unsigned work;

public:
    bool enableCurfew = false;
    unsigned curfewBegin=0;
    unsigned curfewEnd=0;
    bool curfewTimeConverted = false;
    unsigned schoolAgeRestriction = 99;
    bool holidayModeActive = false;
    // add program parameters if we need any, this function got called already
    // from Simulation
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("trace",
            "Trace movements of agent",
            cxxopts::value<unsigned>()->default_value(
                std::to_string(std::numeric_limits<unsigned>::max())))("quarantinePolicy",
            "Quarantine policy: 0 - None, 1 - Agent only, 2 - Agent and "
            "household, 3 - + classroom/work, 4 - + school",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(3))))
            ("quarantineLength",
            "Length of quarantine in days",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(10))));
    }
    void initializeArgs(const cxxopts::ParseResult& result) {
        tracked = result["trace"].as<unsigned>();
        quarantinePolicy = result["quarantinePolicy"].as<unsigned>();
        quarantineLength = result["quarantineLength"].as<unsigned>();
    }
    void init(const parser::LocationTypes& data, unsigned cemeteryID) {
        publicSpace = data.publicSpace;
        home = data.home;
        hospital = data.hospital;
        cemeteryLoc = cemeteryID;
        doctor = data.doctor;
        school = data.school;
        work = data.work;
        classroom = data.classroom;
    }

    void planLocations(Timehandler simTime, unsigned timeStep) {
        //PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();
        unsigned numberOfLocations = realThis->locs->locationListOffsets.size()-1;
        unsigned timestamp = simTime.getTimestamp();
        if (stepsUntilMove.size() == 0) {
            stepsUntilMove.resize(numberOfAgents);
            noWorkLoc.resize(numberOfLocations);
            noWorkAgent.resize(numberOfAgents);
        }
        if (curfewTimeConverted == false) {
            curfewBegin = curfewBegin/timeStep;
            curfewEnd = curfewEnd/timeStep;
            curfewTimeConverted = true;
        }
        //If curfew, noone moves before curfew ends
        thrust::fill(stepsUntilMove.begin(), stepsUntilMove.end(), curfewEnd);
        thrust::fill(realThis->agents->stayedHome.begin(),realThis->agents->stayedHome.end(),true);

        //For each agent that is under 14 years, check if quarantined or school closed, if so flag home as noWork
        thrust::fill(noWorkLoc.begin(), noWorkLoc.end(), (uint8_t)0u);
        thrust::fill(noWorkAgent.begin(), noWorkAgent.end(), (uint8_t)0u);
        unsigned* noWorkLocPtr = thrust::raw_pointer_cast(noWorkLoc.data());
        uint8_t* noWorkAgentPtr = thrust::raw_pointer_cast(noWorkAgent.data());
        thrust::device_vector<typename SimulationType::AgentMeta_t>& agentMetaData = 
            realThis->agents->agentMetaData;
        typename SimulationType::AgentMeta_t *agentMetaDataPtr = thrust::raw_pointer_cast(agentMetaData.data());
        thrust::device_vector<bool>& quarantined = realThis->agents->quarantined;
        bool *quarantinedPtr = thrust::raw_pointer_cast(quarantined.data());
        thrust::device_vector<bool>& locationStates = realThis->locs->states;
        bool *locationStatesPtr = thrust::raw_pointer_cast(locationStates.data());
        thrust::device_vector<unsigned>& closedUntil = realThis->locs->closedUntil;
        unsigned *closedUntilPtr = thrust::raw_pointer_cast(closedUntil.data());
        thrust::device_vector<unsigned long>& locationOffset = realThis->agents->locationOffset;
        unsigned long *locationOffsetPtr = thrust::raw_pointer_cast(locationOffset.data());
        thrust::device_vector<unsigned>& possibleLocations = realThis->agents->possibleLocations;
        unsigned *possibleLocationsPtr = thrust::raw_pointer_cast(possibleLocations.data());
        thrust::device_vector<unsigned>& possibleTypes = realThis->agents->possibleTypes;
        unsigned *possibleTypesPtr = thrust::raw_pointer_cast(possibleTypes.data());
        thrust::device_vector<typename SimulationType::PPState_t>& agentStates =
            realThis->agents->PPValues;
        typename SimulationType::PPState_t *agentStatesPtr = thrust::raw_pointer_cast(agentStates.data());
        thrust::device_vector<AgentStats>& agentStats = realThis->agents->agentStats;
        AgentStats *agentStatsPtr = thrust::raw_pointer_cast(agentStats.data());
        thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locationTypes =
            realThis->locs->locType;
        thrust::device_vector<unsigned>& locationQuarantineUntil = realThis->locs->quarantineUntil;
            unsigned *locationQuarantineUntilPtr = thrust::raw_pointer_cast(locationQuarantineUntil.data());

        //if quarantinePolicy >3, check if more than 1 classrrom in a school is quarantined, if so, quarantine the schooland all classrooms
        if (quarantinePolicy>3) {
            thrust::device_vector<unsigned>& schools = realThis->locs->schools;
            unsigned *schoolsPtr = thrust::raw_pointer_cast(schools.data());
            thrust::device_vector<unsigned>& classrooms = realThis->locs->classrooms;
            unsigned *classroomsPtr = thrust::raw_pointer_cast(classrooms.data());
            thrust::device_vector<unsigned>& classroomOffsets = realThis->locs->classroomOffsets;
            unsigned *classroomOffsetsPtr = thrust::raw_pointer_cast(classroomOffsets.data());
            unsigned numSchools = schools.size();

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
        #pragma omp parallel for
        for (unsigned i = 0; i < numSchools; i++) { RealMovementOps::checkSchoolQuarantine(i, schoolsPtr, classroomsPtr, classroomOffsetsPtr, locationQuarantineUntilPtr, timestamp); }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        RealMovementOps::checkSchoolQuarantineDriver<<<(numSchools - 1) / 256 + 1, 256>>>(
            numSchools, schoolsPtr, classroomsPtr, classroomOffsetsPtr, locationQuarantineUntilPtr, timestamp);
        cudaDeviceSynchronize();
#endif
        }

        //put all workers of quarantined workpalces into quarantine, and clear workplace quarantine
        //put all attendees of quarantined classroom into quarantine, clear classroom quarantine, close classroom 
        if (quarantinePolicy>2) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
            #pragma omp parallel for
            for (unsigned i = 0; i < numberOfAgents; i++) { RealMovementOps::checkSchoolWorkQuarantine(i, agentStatsPtr, agentStatesPtr,
                            quarantinedPtr, locationQuarantineUntilPtr, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr, home, work, school, classroom, timestamp, timeStep, tracked); }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
            RealMovementOps::checkSchoolWorkQuarantineDriver<<<(numberOfAgents - 1) / 256 + 1, 256>>>(
                numberOfAgents, agentStatsPtr, agentStatesPtr, quarantinedPtr, locationQuarantineUntilPtr, locationOffsetPtr, 
                possibleLocationsPtr, possibleTypesPtr, home, work, school, classroom, timestamp, timeStep, tracked);
            cudaDeviceSynchronize();
#endif
            unsigned schoolType = school; unsigned classroomType = classroom; unsigned workType = work;
            //Clear quarantine flags, set closedUntil instead for schools/classrooms
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                                closedUntil.begin(), locationQuarantineUntil.begin(), locationTypes.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(
                                closedUntil.end(), locationQuarantineUntil.end(), locationTypes.end())),
                            [schoolType,classroomType,workType,timestamp]HD(thrust::tuple<unsigned &, unsigned &, typename SimulationType::TypeOfLocation_t&> tup) {
                                if (thrust::get<1>(tup)>timestamp && (thrust::get<2>(tup)==schoolType || thrust::get<2>(tup)==classroomType)) {
                                    thrust::get<0>(tup) = thrust::get<1>(tup);
                                    thrust::get<1>(tup) = 0;
                                } else if (thrust::get<1>(tup)>timestamp && thrust::get<2>(tup)==workType) {
                                    thrust::get<1>(tup) = 0;
                                }
                            }
            );
        }

    //Check if minors are alone at home, flag home
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
        #pragma omp parallel for
        for (unsigned i = 0; i < numberOfAgents; i++) { RealMovementOps::checkUnderageAtHome(i, noWorkLocPtr, agentMetaDataPtr,
                        quarantinedPtr, locationStatesPtr, closedUntilPtr, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr, home, school, classroom,timestamp,schoolAgeRestriction); }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        RealMovementOps::checkUnderageAtHomeDriver<<<(numberOfAgents - 1) / 256 + 1, 256>>>(
            numberOfAgents, noWorkLocPtr, agentMetaDataPtr, quarantinedPtr, locationStatesPtr, closedUntilPtr, locationOffsetPtr, 
            possibleLocationsPtr, possibleTypesPtr, home, school, classroom,timestamp,schoolAgeRestriction);
        cudaDeviceSynchronize();
#endif

        //For each adult working agent (25-65), if home is flagged, at least one adult is flagged as not working that day
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
        #pragma omp parallel for
        for (unsigned i = 0; i < numberOfAgents; i++) { RealMovementOps::setNoWorkToday(i, noWorkLocPtr, noWorkAgentPtr, agentMetaDataPtr,
                        locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr, home); }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        RealMovementOps::setNoWorkTodayDriver<<<(numberOfAgents - 1) / 256 + 1, 256>>>(
            numberOfAgents, noWorkLocPtr, noWorkAgentPtr, agentMetaDataPtr, locationOffsetPtr, 
            possibleLocationsPtr, possibleTypesPtr, home);
        cudaDeviceSynchronize();
#endif
    }

    void movement(Timehandler simTime, unsigned timeStep) {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);

        RealMovementOps::MovementArguments<typename SimulationType::PPState_t,
            typename SimulationType::AgentMeta_t,
            typename SimulationType::TypeOfLocation_t>
            a;

        a.quarantinePolicy = quarantinePolicy;
        a.quarantineLength = quarantineLength;
        a.tracked = this->tracked;
        a.hospitalType = hospital;
        a.homeType = home;
        a.publicPlaceType = publicSpace;
        a.doctorType = doctor;
        a.timeStep = timeStep;
        a.simTime = TimeDay(simTime);
        a.cemeteryLoc = cemeteryLoc;
        a.schoolType = school;
        a.classroomType = classroom;
        a.workType = work;
        a.enableCurfew = enableCurfew;
        a.curfewBegin = curfewBegin;
        a.curfewEnd = curfewEnd;
        a.schoolAgeRestriction = schoolAgeRestriction;

        // Location-based data
        thrust::device_vector<unsigned>& locationAgentList = realThis->locs->locationAgentList;
        unsigned* locationAgentListPtr = thrust::raw_pointer_cast(locationAgentList.data());
        thrust::device_vector<unsigned>& locationListOffsets = realThis->locs->locationListOffsets;
        unsigned* locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
        thrust::device_vector<unsigned>& locationIdsOfAgents = realThis->locs->locationIdsOfAgents;
        unsigned* locationIdsOfAgentsPtr = thrust::raw_pointer_cast(locationIdsOfAgents.data());
        thrust::device_vector<bool>& locationStates = realThis->locs->states;
        a.locationStatesPtr = thrust::raw_pointer_cast(locationStates.data());
        thrust::device_vector<unsigned>& closedUntil = realThis->locs->closedUntil;
        a.closedUntilPtr = thrust::raw_pointer_cast(closedUntil.data());
        thrust::device_vector<unsigned>& locationCapacities = realThis->locs->capacity;
        a.locationCapacitiesPtr = thrust::raw_pointer_cast(locationCapacities.data());
        thrust::device_vector<unsigned>& locationQuarantineUntil = realThis->locs->quarantineUntil;
        a.locationQuarantineUntilPtr = thrust::raw_pointer_cast(locationQuarantineUntil.data());
        thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locationTypes =
            realThis->locs->locType;
        a.locationTypePtr = thrust::raw_pointer_cast(locationTypes.data());

        // Agent-based data
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        a.agentLocationsPtr = thrust::raw_pointer_cast(agentLocations.data());
        thrust::device_vector<unsigned>& agentTypes = realThis->agents->types;
        a.agentTypesPtr = thrust::raw_pointer_cast(agentTypes.data());
        thrust::device_vector<typename SimulationType::PPState_t>& agentStates =
            realThis->agents->PPValues;
        a.agentStatesPtr = thrust::raw_pointer_cast(agentStates.data());
        thrust::device_vector<typename SimulationType::AgentMeta_t>& agentMetaData = 
            realThis->agents->agentMetaData;
        a.agentMetaDataPtr = thrust::raw_pointer_cast(agentMetaData.data());
        thrust::device_vector<bool>& diagnosed = realThis->agents->diagnosed;
        a.diagnosedPtr = thrust::raw_pointer_cast(diagnosed.data());
        thrust::device_vector<bool>& stayedHome = realThis->agents->stayedHome;
        a.stayedHomePtr = thrust::raw_pointer_cast(stayedHome.data());
        thrust::device_vector<bool>& quarantined = realThis->agents->quarantined;
        a.quarantinedPtr = thrust::raw_pointer_cast(quarantined.data());
        thrust::device_vector<AgentStats>& agentStats = realThis->agents->agentStats;
        a.agentStatsPtr = thrust::raw_pointer_cast(agentStats.data());
        a.stepsUntilMovePtr = thrust::raw_pointer_cast(this->stepsUntilMove.data());
        a.noWorkAgentPtr = thrust::raw_pointer_cast(noWorkAgent.data());

        // Arrays storing actual location IDs for each agent, for each location
        // type
        thrust::device_vector<unsigned long>& locationOffset = realThis->agents->locationOffset;
        a.locationOffsetPtr = thrust::raw_pointer_cast(locationOffset.data());
        thrust::device_vector<unsigned>& possibleLocations = realThis->agents->possibleLocations;
        a.possibleLocationsPtr = thrust::raw_pointer_cast(possibleLocations.data());
        thrust::device_vector<unsigned>& possibleTypes = realThis->agents->possibleTypes;
        a.possibleTypesPtr = thrust::raw_pointer_cast(possibleTypes.data());

        // Arrays storing movement behaviour with general locationTypes - for
        // each agent type, WB state, and day
        thrust::device_vector<unsigned>& eventOffset = realThis->agents->agentTypes.eventOffset;
        a.eventOffsetPtr = thrust::raw_pointer_cast(eventOffset.data());
        thrust::device_vector<AgentTypeList::Event>& events = realThis->agents->agentTypes.events;
        a.eventsPtr = thrust::raw_pointer_cast(events.data());

        unsigned numberOfAgents = agentLocations.size();
        unsigned numberOfLocations = locationListOffsets.size() - 1;

        a.day = simTime.getDay();
        if (holidayModeActive) {
            a.day = Days::SUNDAY;
        }
        a.timestamp = simTime.getTimestamp();

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
        for (unsigned i = 0; i < numberOfAgents; i++) { RealMovementOps::doMovement(i, a); }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        RealMovementOps::doMovementDriver<<<(numberOfAgents - 1) / 256 + 1, 256>>>(
            numberOfAgents, a);
        cudaDeviceSynchronize();
#endif
        Util::updatePerLocationAgentLists(
            agentLocations, locationIdsOfAgents, locationAgentList, locationListOffsets);
    }
};
