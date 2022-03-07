#pragma once
#include "locationList.h"
#include "util.h"
#include "timeHandler.h"
#include <iostream>
#include <string>

template<class SimulationType>
class BasicInfection {
private:
    double k;
    unsigned dumpToFile = 0;
    double c0_offset;
    int d_offset, d_peak_offset;
    bool flagInfectionAtLocations = false;
    std::string dumpDirectory = "";
    thrust::device_vector<unsigned> newInfectionsAtLocationsAccumulator;
    thrust::device_vector<bool> infectionFlagAtLocations;
    thrust::device_vector<unsigned> newlyInfectedAgents;
    std::ofstream file;
    thrust::device_vector<unsigned> susceptible1;

    class DumpStore {
        std::vector<unsigned> timestamp;
        std::vector<unsigned> variant;
        std::vector<unsigned> locationIds;
        std::vector<unsigned> locationOffsets; //For the nth timeStampVariant
        std::vector<unsigned> infectedAgentsAtLocation;
        std::vector<unsigned> infectedAgentOffsets; //For the nth locationId
        std::vector<unsigned> infectiousAgentsAtLocation;
        std::vector<unsigned> infectiousAgentOffsets; //For the nth locationId
        std::vector<float>    infectiousnessOfAgents;
        public:
        DumpStore() {
            locationOffsets.push_back(0);
            infectedAgentOffsets.push_back(0);
            infectiousAgentOffsets.push_back(0);
        }
        void insert(unsigned _timestamp, unsigned _variant, thrust::host_vector<unsigned> _locationIds,
                    thrust::host_vector<unsigned> _infectedAgentsAtLocation, thrust::host_vector<unsigned> _peopleOffsetsByLocation,
                    thrust::host_vector<unsigned> _infectiousAgents,
                    thrust::host_vector<float> _infectiousnessOfAgents) {
                        timestamp.push_back(_timestamp);
                        variant.push_back(_variant);
                        locationIds.insert(locationIds.end(), _locationIds.begin(), _locationIds.end());
                        locationOffsets.push_back(locationIds.size());
                        infectedAgentsAtLocation.insert(infectedAgentsAtLocation.end(), _infectedAgentsAtLocation.begin(), _infectedAgentsAtLocation.end());
                        infectedAgentOffsets.push_back(infectedAgentsAtLocation.size());
                        unsigned prevsize = infectiousAgentsAtLocation.size();
                        thrust::transform(_peopleOffsetsByLocation.begin(), _peopleOffsetsByLocation.end(), _peopleOffsetsByLocation.begin(), [prevsize](unsigned in){return in+prevsize;});
                        infectiousAgentOffsets.insert(infectiousAgentOffsets.end(),_peopleOffsetsByLocation.begin()+1, _peopleOffsetsByLocation.end());
                        infectiousAgentsAtLocation.insert(infectiousAgentsAtLocation.end(), _infectiousAgents.begin(), _infectiousAgents.end());
                        infectiousnessOfAgents.insert(infectiousnessOfAgents.end(), _infectiousnessOfAgents.begin(), _infectiousnessOfAgents.end());
        }
        void write(std::string fname) {
            std::ofstream file;
            file.open(fname);
            file << timestamp.size() << "\n";
            thrust::copy(timestamp.begin(), timestamp.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            thrust::copy(variant.begin(), variant.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            thrust::copy(locationOffsets.begin(),
                locationOffsets.end(),
                std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            thrust::copy(locationIds.begin(), locationIds.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            thrust::copy(infectedAgentOffsets.begin(), infectedAgentOffsets.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            thrust::copy(infectedAgentsAtLocation.begin(), infectedAgentsAtLocation.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            thrust::copy(infectiousAgentOffsets.begin(), infectiousAgentOffsets.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            thrust::copy(infectiousAgentsAtLocation.begin(), infectiousAgentsAtLocation.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            thrust::copy(infectiousnessOfAgents.begin(), infectiousnessOfAgents.end(), std::ostream_iterator<float>(file, " "));
            file << "\n";
            file.close();
        }
    };
    DumpStore *dumpstore = nullptr;

public:
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("k,infectionCoefficient",
            "Infection: >0 :infectiousness coefficient ",
            cxxopts::value<double>()->default_value("0.000347"))("dumpLocationInfections",// 291642
            "Dump per-location statistics every N timestep ",
            cxxopts::value<unsigned>()->default_value("0"))("dumpLocationInfectiousList",
            "Dump per-location list of infectious people ",
            cxxopts::value<std::string>()->default_value(""))
            ("d_offset",
            "seasonality day offset ",
            cxxopts::value<int>()->default_value("-5"))
            ("d_peak_offset",
            "seasonality peak day offset ",
            cxxopts::value<int>()->default_value("0"))
            ("c0",
            "seasonality c0 value ",
            cxxopts::value<double>()->default_value("3.08"));
    }

    void finalize() {
        if (flagInfectionAtLocations)
            dumpstore->write(dumpDirectory);
        delete dumpstore;
    }

protected:
    void initializeArgs(const cxxopts::ParseResult& result) {
        this->k = result["infectionCoefficient"].as<double>();
        dumpToFile = result["dumpLocationInfections"].as<unsigned>();
        dumpDirectory = result["dumpLocationInfectiousList"].as<std::string>();
        d_offset = result["d_offset"].as<int>();
        d_peak_offset = result["d_peak_offset"].as<int>();
        c0_offset = result["c0"].as<double>();
        flagInfectionAtLocations = (dumpDirectory == "") ? false : true;
        if (flagInfectionAtLocations) dumpstore = new DumpStore();
    }

public:
    template<typename PPState_t>
    void dumpToFileStep1(thrust::device_vector<unsigned>& locationListOffsets,
        thrust::device_vector<unsigned>& locationAgentList,
        thrust::device_vector<PPState_t>& ppstates,
        thrust::device_vector<float>& fullInfectedCounts,
        thrust::device_vector<unsigned>& agentLocations,
        Timehandler& simTime,
        uint8_t variant) {
        if (susceptible1.size() == 0) {
            susceptible1.resize(locationListOffsets.size() - 1, 0);
        } else {
            thrust::fill(susceptible1.begin(), susceptible1.end(), 0u);
        }

        if (newInfectionsAtLocationsAccumulator.size() == 0) {
            newInfectionsAtLocationsAccumulator.resize(locationListOffsets.size() - 1);
            thrust::fill(newInfectionsAtLocationsAccumulator.begin(), newInfectionsAtLocationsAccumulator.end(), (unsigned)0);
        }
        if (dumpToFile > 0) {// Aggregate new infected counts
            reduce_by_location(locationListOffsets,
                locationAgentList,
                susceptible1,
                ppstates,
                agentLocations,
                [variant] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned { return ppstate.getSusceptible(variant) > 0; });
        }
        if (dumpToFile > 0 && simTime.getTimestamp() % dumpToFile == 0) {
            file.open("locationStats_v" + std::to_string(variant) + "_" + std::to_string(simTime.getTimestamp()) + ".txt");
            // Count number of people at each location
            thrust::device_vector<unsigned> location(locationListOffsets.size() - 1);
            thrust::transform(locationListOffsets.begin() + 1,
                locationListOffsets.end(),
                locationListOffsets.begin(),
                location.begin(),
                thrust::minus<unsigned>());
            // Count number of infected people at each locaiton
            thrust::device_vector<unsigned> infectedCount(locationListOffsets.size() - 1, 0);
            reduce_by_location(locationListOffsets,
                locationAgentList,
                infectedCount,
                ppstates,
                agentLocations,
                [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned { return ppstate.isInfectious() > 0; });
            // Print people/location
            thrust::copy(location.begin(), location.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            // Print infected/location
            thrust::copy(infectedCount.begin(), infectedCount.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            // Print weighted infected counts too
            thrust::copy(fullInfectedCounts.begin(), fullInfectedCounts.end(), std::ostream_iterator<float>(file, " "));
            file << "\n";
        }
    }

    template<typename PPState_t>
    void dumpToFileStep2(thrust::device_vector<unsigned>& locationListOffsets,
        thrust::device_vector<unsigned>& locationAgentList,
        thrust::device_vector<PPState_t>& ppstates,
        thrust::device_vector<unsigned>& agentLocations,
        Timehandler& simTime,
        uint8_t variant) {
        if (dumpToFile > 0) {// Finish aggregating number of new infections
            thrust::device_vector<unsigned> susceptible2(locationListOffsets.size() - 1, 0);
            reduce_by_location(locationListOffsets,
                locationAgentList,
                susceptible2,
                ppstates,
                agentLocations,
                [variant] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned { return ppstate.getSusceptible(variant) > 0; });
            thrust::transform(susceptible1.begin(),
                susceptible1.end(),
                susceptible2.begin(),
                susceptible1.begin(),
                thrust::minus<unsigned>());
            thrust::transform(susceptible1.begin(),
                susceptible1.end(),
                newInfectionsAtLocationsAccumulator.begin(),
                newInfectionsAtLocationsAccumulator.begin(),
                thrust::plus<unsigned>());
        }
        if (dumpToFile > 0 && simTime.getTimestamp() % dumpToFile == 0) {
            // Print new infections at location since last timestep
            thrust::copy(newInfectionsAtLocationsAccumulator.begin(),
                newInfectionsAtLocationsAccumulator.end(),
                std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            file.close();
            thrust::fill(newInfectionsAtLocationsAccumulator.begin(), newInfectionsAtLocationsAccumulator.end(), (unsigned)0);
        }
    }

    template<typename PPState_t>
    void dumpLocationInfectiousList(thrust::device_vector<PPState_t>& ppstates,
        thrust::device_vector<unsigned>& agentLocations,
        thrust::device_vector<unsigned>& fullInfectedCounts,
        thrust::device_vector<unsigned>& newlyInfectedAgents,
        thrust::device_vector<unsigned>& numberOfNewInfectionsAtLocations,
        Timehandler& simTime,
        uint8_t variant) {

        thrust::device_vector<unsigned> outLocationIdOffsets(infectionFlagAtLocations.size());
        auto b2u = [] HD(bool flag) -> unsigned { return flag ? 1u : 0u; };
        thrust::exclusive_scan(thrust::make_transform_iterator(infectionFlagAtLocations.begin(), b2u),
            thrust::make_transform_iterator(infectionFlagAtLocations.end(), b2u),
            outLocationIdOffsets.begin());
        unsigned numberOfLocsWithInfections = outLocationIdOffsets[outLocationIdOffsets.size() - 1]
                                              + infectionFlagAtLocations[infectionFlagAtLocations.size() - 1];
        // no new infections inthis timestep, early exit
        if (numberOfLocsWithInfections == 0) return;

        //
        // List of location IDs
        //
        thrust::device_vector<unsigned> outLocationIds(numberOfLocsWithInfections);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(infectionFlagAtLocations.begin(),
                             thrust::make_permutation_iterator(outLocationIds.begin(), outLocationIdOffsets.begin()),
                             thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(infectionFlagAtLocations.end(),
                thrust::make_permutation_iterator(outLocationIds.begin(), outLocationIdOffsets.end()),
                thrust::make_counting_iterator<unsigned>(0) + infectionFlagAtLocations.size())),
            [] HD(thrust::tuple<bool&, unsigned&, unsigned> tup) {
                if (thrust::get<0>(tup))// if infection at this loc
                    thrust::get<1>(tup) = thrust::get<2>(tup);// then save loc ID
            });
        // std::cout << "list of locs ";
        // thrust::copy(outLocationIds.begin(), outLocationIds.end(),
        // std::ostream_iterator<unsigned>(std::cout, " ")); std::cout << "\n";

        //
        // List of people infected
        //
        // counts
        unsigned totalNewInfections = numberOfNewInfectionsAtLocations[numberOfNewInfectionsAtLocations.size() - 1];
        thrust::exclusive_scan(numberOfNewInfectionsAtLocations.begin(),
            numberOfNewInfectionsAtLocations.end(),
            numberOfNewInfectionsAtLocations.begin());
        totalNewInfections += numberOfNewInfectionsAtLocations[numberOfNewInfectionsAtLocations.size() - 1];
        thrust::device_vector<unsigned> outNumberOfNewInfectionsAtLocations(numberOfLocsWithInfections);
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(infectionFlagAtLocations.begin(),
                thrust::make_permutation_iterator(outNumberOfNewInfectionsAtLocations.begin(), outLocationIdOffsets.begin()),
                numberOfNewInfectionsAtLocations.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(infectionFlagAtLocations.end(),
                thrust::make_permutation_iterator(outNumberOfNewInfectionsAtLocations.begin(), outLocationIdOffsets.end()),
                numberOfNewInfectionsAtLocations.end())),
            [] HD(thrust::tuple<bool&, unsigned&, unsigned> tup) {
                if (thrust::get<0>(tup))// if infection at this loc
                    thrust::get<1>(tup) = thrust::get<2>(tup);// then save loc ID
            });
        // std::cout << "new infection counts  by loc";
        // thrust::copy(outNumberOfNewInfectionsAtLocations.begin(),
        // outNumberOfNewInfectionsAtLocations.end(), std::ostream_iterator<unsigned>(std::cout, "
        // ")); std::cout << totalNewInfections << "\n";
        // indexes
        thrust::device_vector<unsigned> newlyInfectedAgentsByLocation(ppstates.size());
        thrust::device_vector<unsigned> newlyInfectedAgentOffsetsByLocation(ppstates.size());
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationAgentList = realThis->locs->locationAgentList;
        thrust::copy(thrust::make_permutation_iterator(newlyInfectedAgents.begin(), locationAgentList.begin()),
            thrust::make_permutation_iterator(newlyInfectedAgents.begin(), locationAgentList.end()),
            newlyInfectedAgentsByLocation.begin());
        thrust::exclusive_scan(newlyInfectedAgentsByLocation.begin(),
            newlyInfectedAgentsByLocation.end(),
            newlyInfectedAgentOffsetsByLocation.begin());
        // std::cout << "newlyInfectedAgentOffsetsByLocation ";
        // thrust::copy(newlyInfectedAgentOffsetsByLocation.begin(),
        // newlyInfectedAgentOffsetsByLocation.end(), std::ostream_iterator<unsigned>(std::cout, "
        // ")); std::cout << "\n";
        unsigned totalNewInfections2 = newlyInfectedAgentOffsetsByLocation[newlyInfectedAgentOffsetsByLocation.size() - 1]
                                       + newlyInfectedAgentsByLocation[newlyInfectedAgentsByLocation.size() - 1];
        if (totalNewInfections != totalNewInfections2) {
            throw CustomErrors("dumpLocationInfectiousList: mismatch between number of new infected calculations");
        }
        thrust::device_vector<unsigned> newlyInfectedPeopleIds(totalNewInfections);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(newlyInfectedAgentsByLocation.begin(),
                             locationAgentList.begin(),
                             thrust::make_permutation_iterator(
                                 newlyInfectedPeopleIds.begin(), newlyInfectedAgentOffsetsByLocation.begin()))),
            thrust::make_zip_iterator(thrust::make_tuple(newlyInfectedAgentsByLocation.end(),
                locationAgentList.end(),
                thrust::make_permutation_iterator(newlyInfectedPeopleIds.begin(), newlyInfectedAgentOffsetsByLocation.end()))),
            [] HD(thrust::tuple<unsigned&, unsigned&, unsigned&> tup) {
                if (thrust::get<0>(tup))// if person should be written
                    thrust::get<2>(tup) = thrust::get<1>(tup);// then write ID there
            });


        //
        // Length of list at each location, scanned
        //
        // std::cout << "infected counts  ";
        // thrust::copy(fullInfectedCounts.begin(), fullInfectedCounts.end(),
        // std::ostream_iterator<unsigned>(std::cout, " ")); std::cout << "\n"; std::cout << "masks
        // "; thrust::copy(infectionFlagAtLocations.begin(), infectionFlagAtLocations.end(),
        // std::ostream_iterator<bool>(std::cout, " ")); std::cout << "\n";
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.begin(), infectionFlagAtLocations.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.end(), infectionFlagAtLocations.end())),
            [] HD(thrust::tuple<unsigned&, bool&> tup) {
                if (thrust::get<1>(tup) == false) thrust::get<0>(tup) = 0;
            });

        // std::cout << "infected counts masked with flag ";
        // thrust::copy(fullInfectedCounts.begin(), fullInfectedCounts.end(),
        // std::ostream_iterator<unsigned>(std::cout, " ")); std::cout << "\n";
        thrust::device_vector<unsigned> locationLengthAll(fullInfectedCounts.size(), 0);
        thrust::exclusive_scan(fullInfectedCounts.begin(), fullInfectedCounts.end(), locationLengthAll.begin());
        thrust::device_vector<unsigned> locationLength(numberOfLocsWithInfections + 1, 0);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(infectionFlagAtLocations.begin(),
                             thrust::make_permutation_iterator(locationLength.begin(), outLocationIdOffsets.begin()),
                             locationLengthAll.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(infectionFlagAtLocations.end(),
                thrust::make_permutation_iterator(locationLength.begin(), outLocationIdOffsets.end()),
                locationLengthAll.end())),
            [] HD(thrust::tuple<bool&, unsigned&, unsigned> tup) {
                if (thrust::get<0>(tup))// if infection at this loc
                    thrust::get<1>(tup) = thrust::get<2>(tup);// then offset
            });
        locationLength[locationLength.size() - 1] =
            locationLengthAll[locationLengthAll.size() - 1] + fullInfectedCounts[fullInfectedCounts.size() - 1];
        // std::cout << "scanned location lengths ";
        // thrust::copy(locationLength.begin(), locationLength.end(),
        // std::ostream_iterator<unsigned>(std::cout, " ")); std::cout << "\n";
        //
        // indexes of people
        //
        thrust::device_vector<unsigned> peopleFlags(ppstates.size(), 0u);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                             peopleFlags.begin(),
                             thrust::make_permutation_iterator(infectionFlagAtLocations.begin(), agentLocations.begin()))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                peopleFlags.end(),
                thrust::make_permutation_iterator(infectionFlagAtLocations.begin(), agentLocations.end()))),
            [] HD(thrust::tuple<PPState_t&, unsigned&, bool&> tup) {
                PPState_t& state = thrust::get<0>(tup);
                unsigned& personFlag = thrust::get<1>(tup);
                bool& locationFlag = thrust::get<2>(tup);
                if (locationFlag && state.isInfectious()) personFlag = 1;
            });
        // std::cout << "peopleFlags ";
        // thrust::copy(peopleFlags.begin(), peopleFlags.end(),
        // std::ostream_iterator<unsigned>(std::cout, " ")); std::cout << "\n";
        // rearrange people flags
        thrust::device_vector<unsigned> peopleFlagsByLocation(peopleFlags.size());
        thrust::device_vector<unsigned> peopleOffsetsByLocation(peopleFlags.size());
        thrust::copy(thrust::make_permutation_iterator(peopleFlags.begin(), locationAgentList.begin()),
            thrust::make_permutation_iterator(peopleFlags.begin(), locationAgentList.end()),
            peopleFlagsByLocation.begin());
        // std::cout << "peopleFlagsByLocation ";
        // thrust::copy(peopleFlagsByLocation.begin(), peopleFlagsByLocation.end(),
        // std::ostream_iterator<unsigned>(std::cout, " ")); std::cout << "\n";
        thrust::exclusive_scan(peopleFlagsByLocation.begin(), peopleFlagsByLocation.end(), peopleOffsetsByLocation.begin());
        unsigned numberOfPeople = peopleOffsetsByLocation[peopleOffsetsByLocation.size() - 1]
                                  + peopleFlagsByLocation[peopleFlagsByLocation.size() - 1];
        if (numberOfPeople != locationLength[locationLength.size() - 1]) {
            throw CustomErrors("dumpLocationInfectiousList: mismatch between number of people calculations");
        }
        //save infectious people's IDs
        thrust::device_vector<unsigned> peopleIds(numberOfPeople);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(peopleFlagsByLocation.begin(),
                             locationAgentList.begin(),
                             thrust::make_permutation_iterator(peopleIds.begin(), peopleOffsetsByLocation.begin()))),
            thrust::make_zip_iterator(thrust::make_tuple(peopleFlagsByLocation.end(),
                locationAgentList.end(),
                thrust::make_permutation_iterator(peopleIds.begin(), peopleOffsetsByLocation.end()))),
            [] HD(thrust::tuple<unsigned&, unsigned&, unsigned&> tup) {
                if (thrust::get<0>(tup))// if person should be written
                    thrust::get<2>(tup) = thrust::get<1>(tup);// then write ID there
            });

        //save infectious people's infectiousness
        thrust::device_vector<float> infectiousness(numberOfPeople);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(peopleFlagsByLocation.begin(),
                             thrust::make_permutation_iterator(ppstates.begin(),locationAgentList.begin()),
                             thrust::make_permutation_iterator(infectiousness.begin(), peopleOffsetsByLocation.begin()))),
            thrust::make_zip_iterator(thrust::make_tuple(peopleFlagsByLocation.end(),
                thrust::make_permutation_iterator(ppstates.begin(),locationAgentList.end()),
                thrust::make_permutation_iterator(infectiousness.begin(), peopleOffsetsByLocation.end()))),
            [] HD(thrust::tuple<unsigned&, PPState_t&, float&> tup) {
                if (thrust::get<0>(tup))// if person should be written
                    thrust::get<2>(tup) = thrust::get<1>(tup).isInfectious();// then infectiousness
            });

        /*std::ofstream file;
        file.open(dumpDirectory + "infectiousList_v" + std::to_string(variant) + "_" + std::to_string(simTime.getTimestamp())
                  + ".txt");
        file << numberOfLocsWithInfections << "\n";
        thrust::copy(outLocationIds.begin(), outLocationIds.end(), std::ostream_iterator<unsigned>(file, " "));
        file << "\n";
        thrust::copy(outNumberOfNewInfectionsAtLocations.begin(),
            outNumberOfNewInfectionsAtLocations.end(),
            std::ostream_iterator<unsigned>(file, " "));
        file << totalNewInfections << "\n";
        thrust::copy(newlyInfectedPeopleIds.begin(), newlyInfectedPeopleIds.end(), std::ostream_iterator<unsigned>(file, " "));
        file << "\n";
        thrust::copy(locationLength.begin(), locationLength.end(), std::ostream_iterator<unsigned>(file, " "));
        file << "\n";
        thrust::copy(peopleIds.begin(), peopleIds.end(), std::ostream_iterator<unsigned>(file, " "));
        file << "\n";
        file.close();*/
        dumpstore->insert(simTime.getTimestamp(), variant, outLocationIds,
                    newlyInfectedPeopleIds, locationLength, peopleIds,
                    infectiousness);
    }

    double seasonalityMultiplier(Timehandler& simTime) {
        unsigned simDay = simTime.getTimestamp()/simTime.getStepsPerDay();
        //if (simDay<100) return 1.0;
        int d = simTime.getStartDate() + simDay; //267 sept 23
        //int d_peak = 30; //assuming origin = feb 1
        //int d_peak = 45; //assuming origin = feb 15
        // int d_peak = 59; //assuming origin = marc 1
        int d_peak = 59+21+6; //assuming origin = mar 27
        // int d_peak = 90;
        double c0 = 3.08; //0.8;

        if (simDay > 250 && simDay < 450) {
            d += d_offset;
            d_peak += d_peak_offset;
            c0 = c0_offset;
        }
        int d_mod = d % 366;
        if (d_mod > 366 / 2)
            d_mod = 366 - d_mod;
        double normed_value = 
            (0.5 * c0 * cos (2.0 * M_PI * (double)(d_mod - d_peak)/366.0) + (1.0 - 0.5 * c0))/
            (0.5 * c0 * cos (2.0 * M_PI * (double)(d_peak - d_peak)/366.0) + (1.0 - 0.5 * c0));
        double calc_val = std::min(std::max(normed_value, 0.5/*trunc_val*/), 1.0);
        if (d_mod < d_peak)
            return 1.0;
        else
            return calc_val;
    }


    void infectionsAtLocations(Timehandler& simTime, unsigned timeStep, uint8_t variant) {
        //        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationListOffsets =
            realThis->locs->locationListOffsets;// offsets into locationAgentList and locationIdsOfAgents
        thrust::device_vector<unsigned>& locationAgentList = realThis->locs->locationAgentList;
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;

        auto& ppstates = realThis->agents->PPValues;
        auto& infectiousness = realThis->locs->infectiousness;

        thrust::device_vector<double> infectionRatios(locationListOffsets.size() - 1, 0.0);
        thrust::device_vector<float> fullInfectedCounts(locationListOffsets.size() - 1, 0);


        if (flagInfectionAtLocations) {
            if (infectionFlagAtLocations.size() == 0) infectionFlagAtLocations.resize(infectiousness.size());
            thrust::fill(infectionFlagAtLocations.begin(), infectionFlagAtLocations.end(), false);
            if (newlyInfectedAgents.size() == 0) newlyInfectedAgents.resize(ppstates.size());
            thrust::fill(newlyInfectedAgents.begin(), newlyInfectedAgents.end(), 0u);
        }

        //
        // Step 1 - Count up infectious people - those who are Infectious
        //
        reduce_by_location(locationListOffsets,
            locationAgentList,
            fullInfectedCounts,
            ppstates,
            agentLocations,
            [variant] HD(const typename SimulationType::PPState_t& ppstate) -> float { return ppstate.isInfectious(variant); });

        dumpToFileStep1(locationListOffsets, locationAgentList, ppstates, fullInfectedCounts, agentLocations, simTime, variant);

        //
        // Step 2 - calculate infection ratios, based on density of infected people
        //
        double tmpK = this->k;
        double seasonality = seasonalityMultiplier(simTime);
        //if (simTime.getStepsUntilMidnight()==1) printf("%g\n", seasonality);
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.begin(),
                              locationListOffsets.begin(),
                              locationListOffsets.begin() + 1,
                              infectiousness.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                fullInfectedCounts.end(), locationListOffsets.end() - 1, locationListOffsets.end(), infectiousness.end())),
            infectionRatios.begin(),
            [=] HD(thrust::tuple<float&, unsigned&, unsigned&, double&> tuple) {
                float numInfectedAgentsPresent = thrust::get<0>(tuple);
                unsigned offset0 = thrust::get<1>(tuple);
                unsigned offset1 = thrust::get<2>(tuple);
                unsigned num_agents = offset1 - offset0;
                if (numInfectedAgentsPresent == 0.0) { return 0.0; }
                double densityOfInfected = numInfectedAgentsPresent / num_agents;
                /*double y =
                    1.0
                    / (1.0
                        + parTMP.v
                              * std::exp(-parTMP.s * 2 * (densityOfInfected - parTMP.h - 0.5)));
                y = parTMP.a * y + parTMP.b;
                y *= thrust::get<3>(tuple);// Weighted by infectiousness
                return y / (60.0 * 24.0 / static_cast<double>(timeStep));*/
                return 1.0 - exp(-tmpK * densityOfInfected * thrust::get<3>(tuple) * seasonality * static_cast<double>(timeStep));
            });

        //
        // Step 3 - randomly infect susceptible people
        //

        LocationsList<SimulationType>::infectAgents(infectionRatios,
            agentLocations,
            infectionFlagAtLocations,
            newlyInfectedAgents,
            flagInfectionAtLocations,
            simTime,
            variant);
        if (flagInfectionAtLocations) {
            thrust::device_vector<unsigned> fullInfectedCounts2(fullInfectedCounts.size(), 0);
            reduce_by_location(locationListOffsets,
                locationAgentList,
                fullInfectedCounts2,
                ppstates,
                agentLocations,
                [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned {
                    return unsigned(ppstate.isInfectious() > 0);
                });
            thrust::device_vector<unsigned> numberOfNewInfectionsAtLocations(fullInfectedCounts.size(), 0);
            reduce_by_location(locationListOffsets,
                locationAgentList,
                numberOfNewInfectionsAtLocations,
                newlyInfectedAgents,
                agentLocations,
                [] HD(const unsigned& flag) -> unsigned { return flag; });
            dumpLocationInfectiousList(ppstates,
                agentLocations,
                fullInfectedCounts2,
                newlyInfectedAgents,
                numberOfNewInfectionsAtLocations,
                simTime,
                variant);
        }

        dumpToFileStep2(locationListOffsets, locationAgentList, ppstates, agentLocations, simTime, variant);
    }
};
