#pragma once
#include <iostream>
#include "timeHandler.h"
#include "datatypes.h"
#include "cxxopts.hpp"
#include "operators.h"
#include "locationTypesFormat.h"

template<typename SimulationType>
class NoTesting {
public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data) {}

    void performTests(Timehandler simTime, unsigned timeStep) {}
    auto getStats() {return thrust::make_tuple(0u,0u,0u);}
};



namespace DetailedTestingOps {
        template<typename PPState, typename LocationType>
    struct TestingArguments {
        HD TestingArguments() {}
        PPState *agentStatesPtr;
        AgentStats* agentStatsPtr;
        unsigned long* locationOffsetPtr;
        unsigned* possibleLocationsPtr;
        unsigned* possibleTypesPtr;
        unsigned* locationQuarantineUntilPtr;
        unsigned hospitalType;
        unsigned homeType;
        unsigned publicPlaceType;
        unsigned doctorType;
        unsigned schoolType;
        unsigned classroomType;
        unsigned nurseryhomeType;
        unsigned workType;
        unsigned timeStep;
        unsigned timestamp;
        unsigned tracked;
        LocationType* locationTypePtr;
        unsigned *lastTestPtr;
        bool *locationFlagsPtr;
        bool* diagnosedPtr;
        double testingRandom;
        double testingHome;
        double testingWork;
        double testingSchool;
        double testingRandomHospital;
        double testingNurseryHome;
        unsigned testingDelay;
        unsigned quarantineLength;
        bool usePCR;
    };

    template<typename PPState, typename LocationType>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
    void
    flagLocations(unsigned i,  TestingArguments<PPState, LocationType> &a) {
        //If diagnosed in the last 24 hours
        if (a.agentStatsPtr[i].diagnosedTimestamp > a.timestamp - 24 * 60 / a.timeStep) {
            //Mark home
            unsigned home = RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                        a.homeType, a.schoolType, a.workType, 0, nullptr);
            if (home != std::numeric_limits<unsigned>::max())
                a.locationFlagsPtr[home] = true;
            //Mark work
            unsigned work = RealMovementOps::findActualLocationForType(i, a.workType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                        a.homeType, a.schoolType, a.workType, 0, nullptr);
            if (work != std::numeric_limits<unsigned>::max() &&
                (a.locationQuarantineUntilPtr[work] == 0 || //Should test if it was not quarantined, OR
                    (a.locationQuarantineUntilPtr[work] != 0 && //It has been quarantined - either in last 24 hours, OR it's already over
                     (a.locationQuarantineUntilPtr[work] - a.quarantineLength * 24 * 60 / a.timeStep >= a.timestamp - 24 * 60/a.timeStep ||
                      a.locationQuarantineUntilPtr[work] < a.timestamp))))
                a.locationFlagsPtr[work] = true;
            //Mark school
            unsigned school = RealMovementOps::findActualLocationForType(i, a.schoolType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                        a.homeType, a.schoolType, a.workType, 0, nullptr);
            unsigned classroom = std::numeric_limits<unsigned>::max();
            if (school != std::numeric_limits<unsigned>::max() &&
                (a.locationQuarantineUntilPtr[school] == 0 || //Should test if it was not quarantined, OR
                    (a.locationQuarantineUntilPtr[school] != 0 && //It has been quarantined - either in last 24 hours, OR it's already over
                     (a.locationQuarantineUntilPtr[school] - a.quarantineLength * 24 * 60 / a.timeStep >= a.timestamp - 24 * 60/a.timeStep ||
                      a.locationQuarantineUntilPtr[school] < a.timestamp)))) {
                a.locationFlagsPtr[school] = true;
                //Mark classroom too
                classroom = RealMovementOps::findActualLocationForType(i, a.classroomType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                        a.homeType, a.schoolType, a.workType, 0, nullptr);
                if (classroom != std::numeric_limits<unsigned>::max() &&
                    (a.locationQuarantineUntilPtr[classroom] == 0 || //Should test if it was not quarantined, OR
                    (a.locationQuarantineUntilPtr[classroom] != 0 && //It has been quarantined - either in last 24 hours, OR it's already over
                     (a.locationQuarantineUntilPtr[classroom] - a.quarantineLength * 24 * 60 / a.timeStep >= a.timestamp - 24 * 60/a.timeStep ||
                      a.locationQuarantineUntilPtr[classroom] < a.timestamp))))
                    a.locationFlagsPtr[classroom] = true;

            }

            if (a.tracked == i) {
                printf("Testing: Agent %d was diagnosed in last 24 hours, marking home %d, work %d school %d classroom %d\n",
                    i, home, 
                    work==std::numeric_limits<unsigned>::max()?-1:(int)work,
                    school==std::numeric_limits<unsigned>::max()?-1:(int)school,
                    classroom==std::numeric_limits<unsigned>::max()?-1:(int)classroom);
            }
        }

    }
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    template<typename PPState, typename LocationType>
    __global__ void flagLocationsDriver(TestingArguments<PPState, LocationType> a, unsigned numberOfAgents ) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numberOfAgents) { DetailedTestingOps::flagLocations(i, a); }
    }
#endif

template<typename PPState, typename LocationType>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
    void
    doTesting(unsigned i,  TestingArguments<PPState, LocationType> &a) {
        //if recently tested, don't test again
        if (a.timestamp > a.testingDelay*24*60/a.timeStep && 
            a.lastTestPtr[i] != std::numeric_limits<unsigned>::max() &&
            a.lastTestPtr[i] > a.timestamp - a.testingDelay*24*60/a.timeStep) return;
    
        //Check home
        unsigned home = RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                        a.homeType, a.schoolType, a.workType, 0, nullptr);
        bool homeFlag = false;
        if (home != std::numeric_limits<unsigned>::max())
            homeFlag = a.locationFlagsPtr[home];
        //Check work
        unsigned work = RealMovementOps::findActualLocationForType(i, a.workType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                        a.homeType, a.schoolType, a.workType, 0, nullptr);
        bool workFlag = false;
        if (work != std::numeric_limits<unsigned>::max())
            workFlag = a.locationFlagsPtr[work];
        //Check school
        unsigned school = RealMovementOps::findActualLocationForType(i, a.schoolType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                        a.homeType, a.schoolType, a.workType, 0, nullptr);
        unsigned classroom = std::numeric_limits<unsigned>::max();
        bool schoolFlag = false;
        bool classroomFlag = false;
        if (school != std::numeric_limits<unsigned>::max()) {
            schoolFlag = a.locationFlagsPtr[school];
            classroom = RealMovementOps::findActualLocationForType(i, a.classroomType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr,
                                                                        a.homeType, a.schoolType, a.workType, 0, nullptr);
            if (classroom != std::numeric_limits<unsigned>::max())
                classroomFlag = true;
        }

        double testingProbability = a.testingRandom;
        testingProbability += homeFlag * a.testingHome;
        testingProbability += workFlag * a.testingWork;
        testingProbability += schoolFlag * a.testingSchool;
        testingProbability += classroomFlag * 3.0 * a.testingSchool;

        //If agent works in hospital or doctor's office
        if (work != std::numeric_limits<unsigned>::max() &&
            (a.locationTypePtr[work] ==  a.doctorType || 
            a.locationTypePtr[work] ==  a.hospitalType)) {
            testingProbability += a.testingRandomHospital;
        }

        //If agent works in nursery home
        if (work != std::numeric_limits<unsigned>::max() &&
            a.locationTypePtr[work] ==  a.nurseryhomeType) {
            testingProbability += a.testingNurseryHome;
        }

        //If agent is hospitalized for non-COVID
        if (a.agentStatsPtr[i].hospitalizedTimestamp <= a.timestamp &&
            a.agentStatsPtr[i].hospitalizedUntilTimestamp > a.timestamp) {
                
            testingProbability += a.testingRandomHospital;
        }

        if (a.tracked == i && testingProbability>0.0) 
            printf("Testing: Agent %d testing probability: %g\n",
                    i, testingProbability);
        
        //Do the test
        if (testingProbability>1.0 ||
            RandomGenerator::randomReal(1.0) < testingProbability) { 
            a.lastTestPtr[i] = a.timestamp;
            if (a.agentStatesPtr[i].isInfected()) {
                float probability = a.usePCR ? a.agentStatesPtr[i].getAccuracyPCR() : a.agentStatesPtr[i].getAccuracyAntigen();
                if (probability > RandomGenerator::randomReal(1.0)) {
                    a.diagnosedPtr[i] = true;
                    a.agentStatsPtr[i].diagnosedTimestamp = a.timestamp;
                    if (a.tracked == i) 
                        printf("\t Agent %d tested positive\n", i);
                } else {
                    if (a.tracked == i) 
                        printf("\t Agent %d tested FALSE negative\n", i);
                }
            } else {
                //Release from quarantine if home is not quarantined
                if (a.agentStatsPtr[i].quarantinedUntilTimestamp > a.timestamp &&
                    (home != std::numeric_limits<unsigned>::max() && a.locationQuarantineUntilPtr[home] < a.timestamp)) {
                    //Reduce number of days spent in quarantine
                    if (a.agentStatsPtr[i].daysInQuarantine > 0)
                        a.agentStatsPtr[i].daysInQuarantine -= (a.agentStatsPtr[i].quarantinedUntilTimestamp - a.timestamp)/(24*60/a.timeStep);
                    //End quarantine
                    a.agentStatsPtr[i].quarantinedUntilTimestamp = a.timestamp;//a.quarantinedPtr will be cleared by next movementPolicy
                }
                if (a.tracked == i) 
                    printf("\t Agent %d tested negative\n", i);
            }
            
        } else if (testingProbability>0.0) {
            if (a.tracked == i) 
                printf("\t Agent %d was not tested\n", i);
        }

        
    }
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    template<typename PPState, typename LocationType>
    __global__ void doTestingDriver(TestingArguments<PPState, LocationType> a, unsigned numberOfAgents ) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numberOfAgents) { DetailedTestingOps::doTesting(i, a); }
    }
#endif
}// namespace DetailedTestingOps


template<typename SimulationType>
class DetailedTesting {
    unsigned publicSpace;
    unsigned home;
    unsigned hospital;
    unsigned doctor;
    unsigned tracked;
    unsigned quarantineLength;
    unsigned school;
    unsigned classroom;
    unsigned nurseryhome;
    unsigned work;
    thrust::tuple<unsigned, unsigned, unsigned> stats;
    thrust::device_vector<unsigned> lastTest;
    thrust::device_vector<bool> locationFlags;
    double testingRandom = 0.005;
    double testingHome = 0.2;
    double testingWork = 0.1;
    double testingSchool = 0.1;
    double testingRandomHospital = 0.2;
    double testingNurseryHome = 0.3;
    unsigned testingDelay = 5;
    bool usePCR = true;

public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("testingProbabilities",
            "Testing probabilities for random, if someone else was diagnosed at home/work/school, and random for hospital workers: comma-delimited string random,home,work,school,hospital,nurseryHome",
            cxxopts::value<std::string>()->default_value("0.00005,0.01,0.0005,0.0005,0.005,0.05"))
            ("testingRepeatDelay",
            "Minimum number of days between taking tests",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(5))))
            ("testingMethod",
            "default method for testing. Can be PCR (default) on antigen. Accuracies are provided in progression json input",
            cxxopts::value<std::string>()->default_value("PCR"));
    }
    void initializeArgs(const cxxopts::ParseResult& result) {
        testingDelay = result["testingRepeatDelay"].as<unsigned>();
        std::string probsString = result["testingProbabilities"].as<std::string>();
        std::stringstream ss(probsString);
        std::string arg;
        std::vector<double> params;
        for (char i; ss >> i;) {
            arg.push_back(i);    
            if (ss.peek() == ',') {
                if (arg.length()>0 && isdigit(arg[0])) {
                    params.push_back(atof(arg.c_str()));
                    arg.clear();
                }
                ss.ignore();
            }
        }
        if (arg.length()>0 && isdigit(arg[0])) {
            params.push_back(atof(arg.c_str()));
            arg.clear();
        }
        if (params.size()>0) testingRandom = params[0];
        if (params.size()>1) testingHome = params[1];
        if (params.size()>2) testingWork = params[2];
        if (params.size()>3) testingSchool = params[3];
        if (params.size()>4) testingRandomHospital = params[4];
        if (params.size()>5) testingNurseryHome = params[5];
        //printf("testing probabilities: %g %g %g %g %g\n", testingRandom, testingHome, testingWork, testingSchool, testingRandomHospital);
        try {
            quarantineLength = result["quarantineLength"].as<unsigned>();
        } catch (std::exception& e) { quarantineLength = 14; }

        if (result["testingMethod"].as<std::string>().compare("PCR")==0)
            usePCR = true;
        else if (result["testingMethod"].as<std::string>().compare("antigen")==0)
            usePCR = false;
        else throw CustomErrors("unrecognized testingMethod "+result["testingMethod"].as<std::string>()+" must be either PCR or antigen");
    }
    auto getStats() {return stats;}

    void init(const parser::LocationTypes& data) {
        publicSpace = data.publicSpace;
        home = data.home;
        hospital = data.hospital;
        doctor = data.doctor;
        school = data.school;
        work = data.work;
        classroom = data.classroom;
        nurseryhome = data.nurseryhome;
    }

    void performTests(Timehandler simTime, unsigned timeStep) {
        //PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        DetailedTestingOps::TestingArguments<typename SimulationType::PPState_t, typename SimulationType::TypeOfLocation_t> a;

        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfLocations = realThis->locs->locType.size();
        unsigned numberOfAgents = agentLocations.size();
        a.timestamp = simTime.getTimestamp();

        //Running for the first time - initialize arrays
        if (lastTest.size()==0) {
            lastTest.resize(numberOfAgents);
            thrust::fill(lastTest.begin(), lastTest.end(), std::numeric_limits<unsigned>::max());
            locationFlags.resize(numberOfLocations);
            tracked = realThis->locs->tracked;
        }
        //Set all flags of all locations to false (no recent diagnoses)
        thrust::fill(locationFlags.begin(), locationFlags.end(), false);

        a.tracked = tracked;
        a.locationFlagsPtr = thrust::raw_pointer_cast(locationFlags.data());
        a.lastTestPtr = thrust::raw_pointer_cast(lastTest.data());
        a.hospitalType = hospital;
        a.homeType = home;
        a.publicPlaceType = publicSpace;
        a.doctorType = doctor;
        a.timeStep = timeStep;
        a.schoolType = school;
        a.classroomType = classroom;
        a.workType = work;
        a.testingHome = testingHome;
        a.testingWork = testingWork;
        a.testingSchool = testingSchool;
        a.testingRandomHospital = testingRandomHospital;
        a.testingRandom = testingRandom;
        a.testingDelay = testingDelay;
        a.quarantineLength = quarantineLength;
        a.testingNurseryHome = testingNurseryHome;
        a.usePCR = usePCR;
        

        //agent data
        thrust::device_vector<AgentStats>& agentStats = realThis->agents->agentStats;
        a.agentStatsPtr = thrust::raw_pointer_cast(agentStats.data());
        thrust::device_vector<typename SimulationType::PPState_t>& agentStates = realThis->agents->PPValues;
        a.agentStatesPtr = thrust::raw_pointer_cast(agentStates.data());
        thrust::device_vector<bool>& diagnosed = realThis->agents->diagnosed;
        a.diagnosedPtr = thrust::raw_pointer_cast(diagnosed.data());
        //primary location types
        thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locationTypes = realThis->locs->locType;
        a.locationTypePtr = thrust::raw_pointer_cast(locationTypes.data());
        // Arrays storing actual location IDs for each agent, for each location type
        thrust::device_vector<unsigned long>& locationOffset = realThis->agents->locationOffset;
        a.locationOffsetPtr = thrust::raw_pointer_cast(locationOffset.data());
        thrust::device_vector<unsigned>& possibleLocations = realThis->agents->possibleLocations;
        a.possibleLocationsPtr = thrust::raw_pointer_cast(possibleLocations.data());
        thrust::device_vector<unsigned>& possibleTypes = realThis->agents->possibleTypes;
        a.possibleTypesPtr = thrust::raw_pointer_cast(possibleTypes.data());
        thrust::device_vector<unsigned>& locationQuarantineUntil = realThis->locs->quarantineUntil;
        a.locationQuarantineUntilPtr = thrust::raw_pointer_cast(locationQuarantineUntil.data());

        //
        //Step 1 - flag locations of anyone diagnosed yesterday
        //

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
        for (unsigned i = 0; i < numberOfAgents; i++) { DetailedTestingOps::flagLocations(i, a); }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        DetailedTestingOps::flagLocationsDriver<<<(numberOfAgents - 1) / 256 + 1, 256>>>(a, numberOfAgents);
        cudaDeviceSynchronize();
#endif

        //
        //Step 2 - do the testing
        //

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
        for (unsigned i = 0; i < numberOfAgents; i++) { DetailedTestingOps::doTesting(i, a); }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        DetailedTestingOps::doTestingDriver<<<(numberOfAgents - 1) / 256 + 1, 256>>>(a, numberOfAgents);
        cudaDeviceSynchronize();
#endif

        //
        // Step 3 - calculate statistics
        //

        unsigned timestamp = simTime.getTimestamp();
        //Count up those who were tested just now
        unsigned tests = thrust::count(lastTest.begin(), lastTest.end(),timestamp);
        //TODO: count up tests performed in movementPolicy
        //...
        //Count up those who have just been diagnosed because of this testing policy
        unsigned positive1 = thrust::count_if(agentStats.begin(), agentStats.end(), [timestamp] HD (const AgentStats &s){return s.diagnosedTimestamp==timestamp;});
        //Count up those who were diagnosed yesterday, because of a doctor/hospital visit (in movementPolicy)
        unsigned positive2 = thrust::count_if(agentStats.begin(), agentStats.end(), [timestamp,timeStep] HD (const AgentStats &s){return s.diagnosedTimestamp<timestamp && s.diagnosedTimestamp>timestamp-24*60/timeStep;});
        stats = thrust::make_tuple(tests, positive1, positive2);
    }
};
