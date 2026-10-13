#pragma once
#include "locationList.h"
#include "util.h"
#include "timeHandler.h"
#include "randomGenerator.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <map>
#include <sstream>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include "customExceptions.h"

template<class SimulationType>
class BasicInfection {
private:
    double k;
    unsigned dumpToFile = 0;
    double c0_offset, trunc;
    std::vector<int> d_offset;
    int d_peak_offset;
    bool flagInfectionAtLocations = false;
    std::string dumpDirectory = "";
    double sheddingShape = 0.0;
    double traitShape = 0.0;
    std::string traitDist = "gamma";
    unsigned traitPersist = 1;
    double traitSusceptibilityExp = 0.0;
    std::string traitDumpPath = "";
    std::string sheddingMode = "constant";
    bool peakMode = false;
    int peakStartDay = 0;
    int peakEndDay = -1;
    int peakAppliedDay = -1;
    std::string profileOutPath = "";
    std::vector<double> profileSum = std::vector<double>(61, 0.0);
    std::vector<double> profileCount = std::vector<double>(61, 0.0);
    thrust::device_vector<float> traitInfectiousness;
    thrust::device_vector<float> traitSusceptibility;
    bool traitsInitialised = false;
    int profileAppliedDay = -1;
    double densityExponent = 1.0;
    std::string infectionTracePath = "";
    std::string occupancyStatsPath = "";
    std::string volumeModelMode = "off";
    std::string volumeTypeFactors = "";
    unsigned volumeHomeType = 2;
    bool volumeInitialised = false;
    bool volumeFillOne = false;
    bool volumeActive = false;
    thrust::device_vector<float> volumeDenominator;
    thrust::device_vector<unsigned> occupancyDayMax;
    thrust::device_vector<unsigned> occupancyMax;
    std::vector<std::vector<unsigned>> occupancyDailyHistory;
    int occupancyCurrentDay = -1;
    std::string eventLogPath = "";
    std::ofstream eventLogStream;
    int eventLogDay = -1;
    thrust::device_vector<unsigned> eventPresentMax;
    thrust::device_vector<unsigned> eventInfectiousMax;
    std::vector<unsigned> eventTypeByLocation;
    bool initialSheddingDone = false;
    thrust::device_vector<unsigned> newInfectionsAtLocationsAccumulator;
    thrust::device_vector<bool> infectionFlagAtLocations;
    thrust::device_vector<unsigned> newlyInfectedAgents;
    std::ofstream file;
    thrust::device_vector<unsigned> susceptible1;

    thrust::device_vector<float> infectionRatios;
    thrust::device_vector<float> fullInfectedCounts;

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
            cxxopts::value<std::string>()->default_value("-5"))
            ("d_peak_offset",
            "seasonality peak day offset ",
            cxxopts::value<int>()->default_value("0"))
            ("trunc",
            "seasonality truncation value for Omicron ",
            cxxopts::value<double>()->default_value("0.5"))
            ("c0",
            "seasonality c0 value ",
            cxxopts::value<double>()->default_value("3.08"))
            ("sheddingShape",
            "Gamma shape of a per-infection shedding multiplier with mean 1. <=0 disables it",
            cxxopts::value<double>()->default_value("0"))
            ("traitShape",
            "Shape of a persistent per-agent infectiousness trait with mean 1 (gamma CV2 = 1/shape, lognormal likewise). <=0 disables it",
            cxxopts::value<double>()->default_value("0"))
            ("traitDist",
            "Family of the persistent trait: gamma or lognormal",
            cxxopts::value<std::string>()->default_value("gamma"))
            ("traitPersist",
            "1: the trait is drawn once per agent and kept for life",
            cxxopts::value<unsigned>()->default_value("1"))
            ("traitSusceptibilityExp",
            "Exponent rho: susceptibility weight is trait^rho scaled to mean 1 over agents, applied on the hazard scale. 0 disables it",
            cxxopts::value<double>()->default_value("0"))
            ("traitDump",
            "Write per-agent trait, susceptibility weight, age, agent type and home location type to this CSV path",
            cxxopts::value<std::string>()->default_value(""))
            ("sheddingMode",
            "constant: the per-episode multiplier scales infectiousness at all times; peak: only on days since infection inside sheddingPeakWindow",
            cxxopts::value<std::string>()->default_value("constant"))
            ("sheddingPeakWindow",
            "t1,t2: calendar days since the infection day (inclusive) in which the multiplier acts in peak mode",
            cxxopts::value<std::string>()->default_value(""))
            ("profileOut",
            "Write the mean infectiousness by days since infection to this CSV path",
            cxxopts::value<std::string>()->default_value(""))
            ("densityExponent",
            "Infection density is the summed infectiousness divided by occupancy to this power. 1 is the original kernel",
            cxxopts::value<double>()->default_value("1"))
            ("volumeModel",
            "Replace the head-count density denominator by capacity times a per-type factor: off, nonresidential, all, "
            "fillone (head count times the type factor, non-residential)",
            cxxopts::value<std::string>()->default_value("off"))
            ("volumeTypeFactors",
            "Per-type effective removal factors for the volume model as type:factor,type:factor",
            cxxopts::value<std::string>()->default_value(""))
            ("volumeHomeType",
            "Location type id of residences, which stay on the head-count form in nonresidential mode",
            cxxopts::value<unsigned>()->default_value("2"))
            ("occupancyStats",
            "Write per-location occupancy statistics (max, daily maxima percentiles) to this TSV path",
            cxxopts::value<std::string>()->default_value(""))
            ("infectionTrace",
            "Write one row per infection (agent, infector, place, time) to this TSV path",
            cxxopts::value<std::string>()->default_value(""))
            ("venueProfile",
            "Location-type transmission profile: off, covid, or caregiving",
            cxxopts::value<std::string>()->default_value("off"))
            ("venueStrength",
            "Scale of the venue profile, 0 = flat, 1 = full profile",
            cxxopts::value<double>()->default_value("1"))
            ("communityScale",
            "Multiplier on non-home, non-hospital, non-nursing location infectiousness",
            cxxopts::value<double>()->default_value("1"))
            ("eventRate",
            "Daily probability that a social venue draws an amplification event",
            cxxopts::value<double>()->default_value("0"))
            ("eventFactor",
            "Infectiousness multiplier while a social-venue event is active",
            cxxopts::value<double>()->default_value("20"))
            ("eventTypes",
            "Comma-separated location types eligible for events. Default 5,6,8,9",
            cxxopts::value<std::string>()->default_value("5,6,8,9"))
            ("eventLog",
            "Write one row per amplification event (day, location, type, peak presence)",
            cxxopts::value<std::string>()->default_value(""))
            ("homeVenueFactor",
            "If non-negative, replace the venue-profile base factor of type 2 (home)",
            cxxopts::value<double>()->default_value("-1"));
    }

    [[nodiscard]] double getSheddingShape() const { return sheddingShape; }
    [[nodiscard]] const float* getTraitInfectiousnessPtr() const {
        return traitInfectiousness.empty() ? nullptr : thrust::raw_pointer_cast(traitInfectiousness.data());
    }
    [[nodiscard]] bool getPeakMode() const { return peakMode; }

    void finalize() {
        flushEventLog();
        writeInfectionTrace();
        writeOccupancyStats();
        writeProfile();
        if (flagInfectionAtLocations)
            dumpstore->write(dumpDirectory);
        delete dumpstore;
    }

protected:
    void validateTraitOptions(const std::string& window) {
        if (traitShape > 0.0 && sheddingShape > 0.0) {
            throw CustomErrors("--sheddingShape and --traitShape are mutually exclusive");
        }
        if (traitDist != "gamma" && traitDist != "lognormal") {
            throw CustomErrors("--traitDist must be gamma or lognormal");
        }
        if (traitShape > 0.0 && traitPersist != 1) {
            throw CustomErrors("--traitShape needs --traitPersist 1; the per-episode case is --sheddingShape");
        }
        if (traitSusceptibilityExp != 0.0 && !(traitShape > 0.0)) {
            throw CustomErrors("--traitSusceptibilityExp needs --traitShape > 0");
        }
        if (traitSusceptibilityExp < 0.0) {
            throw CustomErrors("--traitSusceptibilityExp must not be negative");
        }
        if (sheddingMode != "constant" && sheddingMode != "peak") {
            throw CustomErrors("--sheddingMode must be constant or peak");
        }
        peakMode = sheddingMode == "peak";
        if (peakMode) {
            if (!(sheddingShape > 0.0)) throw CustomErrors("--sheddingMode peak needs --sheddingShape > 0");
            auto comma = window.find(',');
            if (comma == std::string::npos) throw CustomErrors("--sheddingMode peak needs --sheddingPeakWindow t1,t2");
            peakStartDay = std::stoi(window.substr(0, comma));
            peakEndDay = std::stoi(window.substr(comma + 1));
            if (peakStartDay < 1 || peakEndDay < peakStartDay) {
                throw CustomErrors("--sheddingPeakWindow needs 1 <= t1 <= t2");
            }
        }
    }

    void initializeArgs(const cxxopts::ParseResult& result) {
        this->k = result["infectionCoefficient"].as<double>();
        dumpToFile = result["dumpLocationInfections"].as<unsigned>();
        dumpDirectory = result["dumpLocationInfectiousList"].as<std::string>();
        d_offset = splitStringInt(result["d_offset"].as<std::string>(), ',');
        d_peak_offset = result["d_peak_offset"].as<int>();
        c0_offset = result["c0"].as<double>();
        trunc = result["trunc"].as<double>();
        flagInfectionAtLocations = (dumpDirectory == "") ? false : true;
        sheddingShape = result["sheddingShape"].as<double>();
        traitShape = result["traitShape"].as<double>();
        traitDist = result["traitDist"].as<std::string>();
        traitPersist = result["traitPersist"].as<unsigned>();
        traitSusceptibilityExp = result["traitSusceptibilityExp"].as<double>();
        traitDumpPath = result["traitDump"].as<std::string>();
        sheddingMode = result["sheddingMode"].as<std::string>();
        profileOutPath = result["profileOut"].as<std::string>();
        validateTraitOptions(result["sheddingPeakWindow"].as<std::string>());
        densityExponent = result["densityExponent"].as<double>();
        infectionTracePath = result["infectionTrace"].as<std::string>();
        occupancyStatsPath = result["occupancyStats"].as<std::string>();
        eventLogPath = result["eventLog"].as<std::string>();
        volumeModelMode = result["volumeModel"].as<std::string>();
        volumeTypeFactors = result["volumeTypeFactors"].as<std::string>();
        volumeHomeType = result["volumeHomeType"].as<unsigned>();
        if (flagInfectionAtLocations) dumpstore = new DumpStore();
        if (dumpToFile || flagInfectionAtLocations || !infectionTracePath.empty() || !occupancyStatsPath.empty()
            || volumeModelMode != "off")
            Util::needAgentsSortedByLocation = 1;
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
        auto realThis = static_cast<SimulationType*>(this);
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
            Util::reduce_by_location(locationListOffsets,
                locationAgentList,
                susceptible1,
                ppstates,
                agentLocations,
                realThis->locs->locationIdsOfAgents,
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
            Util::reduce_by_location(locationListOffsets,
                locationAgentList,
                infectedCount,
                ppstates,
                agentLocations,
                realThis->locs->locationIdsOfAgents,
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
            auto realThis = static_cast<SimulationType*>(this);
            Util::reduce_by_location(locationListOffsets,
                locationAgentList,
                susceptible2,
                ppstates,
                agentLocations,
                realThis->locs->locationIdsOfAgents,
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
        // int d_peak = 59+21+6; //assuming origin = mar 27
        int d_peak = 59+22; //assuming origin = mar 22
        d += 6; //offset by 6 //10 better?

        // int d_peak = 90;
        double c0 = 3.08; //0.8;
        double trunc_val = 0.5;

        if (simDay > 250 && simDay < 450) {
            d += d_offset[0];
            d_peak += d_peak_offset;
            c0 = c0_offset;
        } else if (simDay > 600) {
            trunc_val = trunc;
            if (d_offset.size()>0)
                d += d_offset[1];
        }
        int d_mod = d % 366;
        if (d_mod > 366 / 2)
            d_mod = 366 - d_mod;
        double normed_value = 
            (0.5 * c0 * cos (2.0 * M_PI * (double)(d_mod - d_peak)/366.0) + (1.0 - 0.5 * c0))/
            (0.5 * c0 * cos (2.0 * M_PI * (double)(d_peak - d_peak)/366.0) + (1.0 - 0.5 * c0));
        double calc_val = std::min(std::max(normed_value, trunc_val), 1.0);
        if (d_mod < d_peak)
            return 1.0;
        else
            return calc_val;
    }


    static std::uint64_t mixBits(std::uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    void initialiseTraits() {
        if (traitsInitialised) return;
        traitsInitialised = true;
        auto realThis = static_cast<SimulationType*>(this);
        const std::size_t n = realThis->agents->PPValues.size();
        std::vector<float> trait(n, 1.0f);
        std::vector<float> weight(n, 1.0f);
        if (traitShape > 0.0) {
            const std::uint64_t base = RandomGenerator::getBaseSeed();
            const bool lognormal = traitDist == "lognormal";
            const double logVar = std::log(1.0 + 1.0 / traitShape);
            std::vector<double> draws(n);
            for (std::size_t i = 0; i < n; ++i) {
                std::mt19937_64 generator(mixBits(base ^ 0x7472616974ULL) + mixBits(static_cast<std::uint64_t>(i) + 1ULL));
                if (lognormal) {
                    std::lognormal_distribution<double> dist(-0.5 * logVar, std::sqrt(logVar));
                    draws[i] = dist(generator);
                } else {
                    std::gamma_distribution<double> dist(traitShape, 1.0 / traitShape);
                    draws[i] = dist(generator);
                }
                trait[i] = static_cast<float>(draws[i]);
            }
            if (traitSusceptibilityExp > 0.0) {
                double total = 0.0;
                for (std::size_t i = 0; i < n; ++i) total += std::pow(draws[i], traitSusceptibilityExp);
                const double mean = total / static_cast<double>(n);
                for (std::size_t i = 0; i < n; ++i) {
                    weight[i] = static_cast<float>(std::pow(draws[i], traitSusceptibilityExp) / mean);
                }
            }
            traitInfectiousness = trait;
            if (traitSusceptibilityExp > 0.0) traitSusceptibility = weight;
        }
        if (!traitDumpPath.empty()) writeTraitDump(trait, weight);
    }

    void writeTraitDump(const std::vector<float>& trait, const std::vector<float>& weight) {
        auto realThis = static_cast<SimulationType*>(this);
        thrust::host_vector<typename SimulationType::AgentMeta_t> meta = realThis->agents->agentMetaData;
        thrust::host_vector<unsigned> types = realThis->agents->types;
        thrust::host_vector<unsigned long> offsets = realThis->agents->locationOffset;
        thrust::host_vector<unsigned> possibleTypes = realThis->agents->possibleTypes;
        thrust::host_vector<unsigned> possibleLocations = realThis->agents->possibleLocations;
        thrust::host_vector<typename SimulationType::TypeOfLocation_t> locTypes = realThis->locs->locType;
        std::ofstream out(traitDumpPath.c_str());
        out << "agent,trait,susceptibilityWeight,age,agentType,homeLocType\n";
        out.precision(9);
        for (std::size_t i = 0; i < trait.size(); ++i) {
            int homeLocType = -1;
            for (unsigned long j = offsets[i]; j < offsets[i + 1]; ++j) {
                if (possibleTypes[j] == realThis->homeType) {
                    homeLocType = static_cast<int>(locTypes[possibleLocations[j]]);
                    break;
                }
            }
            out << i << ',' << trait[i] << ',' << weight[i] << ',' << static_cast<int>(meta[i].getAge()) << ',' << types[i] << ','
                << homeLocType << '\n';
        }
    }

    void ensureInitialShedding() {
        if (initialSheddingDone) return;
        initialSheddingDone = true;
        initialiseTraits();
        auto realThis = static_cast<SimulationType*>(this);
        double shape = sheddingShape;
        const bool peak = peakMode;
        const float* traitPtr = getTraitInfectiousnessPtr();
        const unsigned unknownTime = 0xffffffffu;
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(realThis->agents->PPValues.begin(),
                             realThis->agents->agentStats.begin(),
                             realThis->agents->location.begin(),
                             thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(realThis->agents->PPValues.end(),
                realThis->agents->agentStats.end(),
                realThis->agents->location.end(),
                thrust::make_counting_iterator<unsigned>(0) + realThis->agents->PPValues.size())),
            [shape, unknownTime, traitPtr, peak] HD(thrust::tuple<typename SimulationType::PPState_t&, AgentStats&, unsigned&, unsigned> tup) {
                auto& state = thrust::get<0>(tup);
                auto& stat = thrust::get<1>(tup);
                unsigned place = thrust::get<2>(tup);
                unsigned agentID = thrust::get<3>(tup);
                if (!state.isInfected()) return;
                if (stat.infectedTimestamp == unknownTime) {
                    stat.infectedTimestamp = 0;
                    stat.infectedLocation = place;
                }
                float multiplier;
                if (traitPtr != nullptr) {
                    multiplier = traitPtr[agentID];
                } else if (shape > 0.0) {
                    multiplier = static_cast<float>(RandomGenerator::gammaMeanOne(shape));
                } else {
                    return;
                }
                state.setShedMultiplier(peak ? 1.0f : multiplier);
                stat.shedMultiplier = multiplier;
            });
    }

    void applyPeakWindow(Timehandler& simTime) {
        if (!peakMode) return;
        const unsigned stepsPerDay = simTime.getStepsPerDay();
        const int day = static_cast<int>(simTime.getTimestamp() / stepsPerDay);
        if (day == peakAppliedDay) return;
        peakAppliedDay = day;
        auto realThis = static_cast<SimulationType*>(this);
        const int t1 = peakStartDay;
        const int t2 = peakEndDay;
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(realThis->agents->PPValues.begin(), realThis->agents->agentStats.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(realThis->agents->PPValues.end(), realThis->agents->agentStats.end())),
            [day, t1, t2, stepsPerDay] HD(thrust::tuple<typename SimulationType::PPState_t&, AgentStats&> tup) {
                auto& state = thrust::get<0>(tup);
                auto& stat = thrust::get<1>(tup);
                if (!state.isInfected() || stat.infectedTimestamp == 0xffffffffu) return;
                int age = day - static_cast<int>(stat.infectedTimestamp / stepsPerDay);
                float target = (age >= t1 && age <= t2) ? stat.shedMultiplier : 1.0f;
                if (state.getShedMultiplier() != target) state.setShedMultiplier(target);
            });
    }

    void accumulateProfile(Timehandler& simTime) {
        if (profileOutPath.empty()) return;
        const unsigned stepsPerDay = simTime.getStepsPerDay();
        const int day = static_cast<int>(simTime.getTimestamp() / stepsPerDay);
        if (day == profileAppliedDay) return;
        profileAppliedDay = day;
        auto realThis = static_cast<SimulationType*>(this);
        thrust::host_vector<typename SimulationType::PPState_t> states = realThis->agents->PPValues;
        thrust::host_vector<AgentStats> stats = realThis->agents->agentStats;
        for (std::size_t i = 0; i < states.size(); ++i) {
            if (!states[i].isInfected() || stats[i].infectedTimestamp == 0xffffffffu || stats[i].infectedTimestamp == 0) continue;
            int age = day - static_cast<int>(stats[i].infectedTimestamp / stepsPerDay);
            if (age < 0 || age >= static_cast<int>(profileSum.size())) continue;
            profileSum[age] += states[i].isInfectious();
            profileCount[age] += 1.0;
        }
    }

    void writeProfile() {
        if (profileOutPath.empty()) return;
        std::ofstream out(profileOutPath.c_str());
        out << "daysSinceInfection,sumInfectiousness,infectedAgents\n";
        out.precision(10);
        for (std::size_t d = 0; d < profileSum.size(); ++d) out << d << ',' << profileSum[d] << ',' << profileCount[d] << '\n';
    }

    void attributeInfectors(uint8_t variant) {
        auto realThis = static_cast<SimulationType*>(this);
        unsigned* offsets = thrust::raw_pointer_cast(realThis->locs->locationListOffsets.data());
        unsigned* people = thrust::raw_pointer_cast(realThis->locs->locationAgentList.data());
        typename SimulationType::PPState_t* states = thrust::raw_pointer_cast(realThis->agents->PPValues.data());
        AgentStats* stats = thrust::raw_pointer_cast(realThis->agents->agentStats.data());
        unsigned* places = thrust::raw_pointer_cast(realThis->agents->location.data());
        unsigned* newly = thrust::raw_pointer_cast(newlyInfectedAgents.data());
        unsigned nAgents = static_cast<unsigned>(realThis->agents->PPValues.size());
        const unsigned unknown = 0xffffffffu;
        const bool keepEpisodeMultiplier = peakMode || traitInfectiousness.size() > 0;
        thrust::for_each(thrust::make_counting_iterator<unsigned>(0),
            thrust::make_counting_iterator<unsigned>(nAgents),
            [=] HD(unsigned agent) {
                if (newly[agent] == 0u) return;
                unsigned place = places[agent];
                float total = 0.0f;
                for (unsigned j = offsets[place]; j < offsets[place + 1]; ++j) {
                    unsigned other = people[j];
                    if (other == agent) continue;
                    total += states[other].isInfectious(variant);
                }
                unsigned infector = unknown;
                if (total > 0.0f) {
                    float draw = static_cast<float>(RandomGenerator::randomUnit()) * total;
                    float cumulative = 0.0f;
                    for (unsigned j = offsets[place]; j < offsets[place + 1]; ++j) {
                        unsigned other = people[j];
                        if (other == agent) continue;
                        cumulative += states[other].isInfectious(variant);
                        if (cumulative >= draw) {
                            infector = other;
                            break;
                        }
                    }
                }
                stats[agent].infectorId = infector;
                if (!keepEpisodeMultiplier) stats[agent].shedMultiplier = states[agent].getShedMultiplier();
                stats[agent].infectorShed = infector != unknown ? stats[infector].shedMultiplier : 1.0f;
                stats[agent].infectorInfectedTimestamp = infector != unknown ? stats[infector].infectedTimestamp : unknown;
                stats[agent].infectedOccupancy = offsets[place + 1] - offsets[place];
                stats[agent].infectiousPresent = total;
            });
    }

    void initialiseVolumeModel() {
        if (volumeInitialised) return;
        volumeInitialised = true;
        if (volumeModelMode == "off") return;
        auto realThis = static_cast<SimulationType*>(this);
        std::map<unsigned, float> factors;
        std::stringstream stream(volumeTypeFactors);
        std::string item;
        while (std::getline(stream, item, ',')) {
            auto colon = item.find(':');
            if (colon == std::string::npos) continue;
            factors[static_cast<unsigned>(std::stoul(item.substr(0, colon)))] = std::stof(item.substr(colon + 1));
        }
        thrust::host_vector<unsigned> capacity = realThis->locs->capacity;
        thrust::host_vector<typename SimulationType::TypeOfLocation_t> types = realThis->locs->locType;
        thrust::host_vector<float> denominator(capacity.size(), 0.0f);
        bool residences = volumeModelMode == "all";
        volumeFillOne = volumeModelMode == "fillone";
        for (unsigned i = 0; i < capacity.size(); ++i) {
            unsigned type = types[i];
            auto found = factors.find(type);
            if (found == factors.end()) continue;
            if (type == volumeHomeType && !residences) continue;
            if (volumeFillOne) denominator[i] = found->second;
            else denominator[i] = static_cast<float>(capacity[i]) * found->second;
        }
        volumeDenominator = denominator;
        volumeActive = true;
    }

    void noteEventExposure() {
        if (eventLogPath.empty()) return;
        auto realThis = static_cast<SimulationType*>(this);
        unsigned n = static_cast<unsigned>(fullInfectedCounts.size());
        if (eventPresentMax.size() != n) {
            eventPresentMax.assign(n, 0u);
            eventInfectiousMax.assign(n, 0u);
        }
        float* boost = thrust::raw_pointer_cast(realThis->locs->eventBoost.data());
        unsigned* offsets = thrust::raw_pointer_cast(realThis->locs->locationListOffsets.data());
        float* infectious = thrust::raw_pointer_cast(fullInfectedCounts.data());
        unsigned* presentMax = thrust::raw_pointer_cast(eventPresentMax.data());
        unsigned* infectiousMax = thrust::raw_pointer_cast(eventInfectiousMax.data());
        thrust::for_each(thrust::make_counting_iterator<unsigned>(0), thrust::make_counting_iterator<unsigned>(n),
            [=] HD(unsigned loc) {
                if (!(boost[loc] > 1.0f)) return;
                unsigned present = offsets[loc + 1] - offsets[loc];
                unsigned infectors = static_cast<unsigned>(infectious[loc]);
                if (present > presentMax[loc]) presentMax[loc] = present;
                if (infectors > infectiousMax[loc]) infectiousMax[loc] = infectors;
            });
    }

    void flushEventLog() {
        if (eventLogPath.empty() || eventLogDay < 0 || eventPresentMax.empty()) return;
        auto realThis = static_cast<SimulationType*>(this);
        thrust::host_vector<float> boost = realThis->locs->eventBoost;
        thrust::host_vector<unsigned> present = eventPresentMax;
        thrust::host_vector<unsigned> infectious = eventInfectiousMax;
        if (eventTypeByLocation.empty()) {
            thrust::host_vector<typename SimulationType::TypeOfLocation_t> types = realThis->locs->locType;
            eventTypeByLocation.assign(types.begin(), types.end());
        }
        if (!eventLogStream.is_open()) {
            eventLogStream.open(eventLogPath.c_str());
            eventLogStream << "day\tlocation_index\tlocType\tmax_present\tmax_infectious\n";
        }
        for (unsigned loc = 0; loc < boost.size(); ++loc) {
            if (!(boost[loc] > 1.0f)) continue;
            unsigned type = loc < eventTypeByLocation.size() ? eventTypeByLocation[loc] : 0u;
            eventLogStream << eventLogDay << '\t' << loc << '\t' << type << '\t'
                           << (loc < present.size() ? present[loc] : 0u) << '\t'
                           << (loc < infectious.size() ? infectious[loc] : 0u) << '\n';
        }
        eventLogStream.flush();
    }

    void trackOccupancy(Timehandler& simTime) {
        if (occupancyStatsPath.empty()) return;
        auto realThis = static_cast<SimulationType*>(this);
        unsigned* offsets = thrust::raw_pointer_cast(realThis->locs->locationListOffsets.data());
        unsigned n = static_cast<unsigned>(realThis->locs->locationListOffsets.size()) - 1;
        if (occupancyMax.size() != n) {
            occupancyMax.assign(n, 0u);
            occupancyDayMax.assign(n, 0u);
        }
        int day = static_cast<int>(simTime.getTimestamp() / simTime.getStepsPerDay());
        if (occupancyCurrentDay >= 0 && day != occupancyCurrentDay) flushOccupancyDay();
        occupancyCurrentDay = day;
        unsigned* dayMax = thrust::raw_pointer_cast(occupancyDayMax.data());
        unsigned* totalMax = thrust::raw_pointer_cast(occupancyMax.data());
        thrust::for_each(thrust::make_counting_iterator<unsigned>(0), thrust::make_counting_iterator<unsigned>(n),
            [=] HD(unsigned loc) {
                unsigned count = offsets[loc + 1] - offsets[loc];
                if (count > dayMax[loc]) dayMax[loc] = count;
                if (count > totalMax[loc]) totalMax[loc] = count;
            });
    }

    void flushOccupancyDay() {
        if (occupancyDayMax.empty()) return;
        thrust::host_vector<unsigned> host = occupancyDayMax;
        occupancyDailyHistory.emplace_back(host.begin(), host.end());
        thrust::fill(occupancyDayMax.begin(), occupancyDayMax.end(), 0u);
    }

    void writeOccupancyStats() {
        if (occupancyStatsPath.empty() || occupancyMax.empty()) return;
        flushOccupancyDay();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::host_vector<typename SimulationType::TypeOfLocation_t> types = realThis->locs->locType;
        thrust::host_vector<unsigned> maxima = occupancyMax;
        std::ofstream out(occupancyStatsPath.c_str());
        out << "location\tlocType\tmaxOccupancy\tdailyMaxMean\tdailyMaxP95\tdaysObserved\n";
        std::vector<unsigned> series;
        for (unsigned loc = 0; loc < maxima.size(); ++loc) {
            series.clear();
            for (const auto& day : occupancyDailyHistory) series.push_back(day[loc]);
            if (series.empty()) continue;
            double mean = 0.0;
            for (unsigned v : series) mean += v;
            mean /= series.size();
            std::sort(series.begin(), series.end());
            unsigned p95 = series[static_cast<size_t>(0.95 * (series.size() - 1))];
            out << loc << '\t' << static_cast<unsigned>(types[loc]) << '\t' << maxima[loc] << '\t' << mean << '\t' << p95 << '\t'
                << series.size() << '\n';
        }
    }

    void writeInfectionTrace() {
        if (infectionTracePath.empty()) return;
        auto realThis = static_cast<SimulationType*>(this);
        thrust::host_vector<AgentStats> stats = realThis->agents->agentStats;
        thrust::host_vector<typename SimulationType::TypeOfLocation_t> types = realThis->locs->locType;
        std::ofstream out(infectionTracePath.c_str());
        if (!out) {
            std::cerr << "Could not write infection trace " << infectionTracePath << "\n";
            return;
        }
        out << "agent\tinfector\tlocation\tlocType\ttimestamp\tvariant\tshed\tdiagnosed\t"
               "hospitalized\tinfectedCount\tworstState\toccupancy\tinfectiousPresent\tinfectorShed\tinfectorTimestamp\t"
               "symptomOnset\tquarantinedFrom\tquarantinedUntil\n";
        const unsigned unknownTime = std::numeric_limits<unsigned>::max();
        for (unsigned agent = 0; agent < stats.size(); ++agent) {
            const AgentStats& stat = stats[agent];
            if (stat.infectedCount == 0 && stat.infectedTimestamp == unknownTime) continue;
            unsigned place = stat.infectedLocation;
            unsigned locType = place < types.size() ? types[place] : 0u;
            out << agent << '\t' << stat.infectorId << '\t' << place << '\t' << locType << '\t'
                << stat.infectedTimestamp << '\t' << stat.variant << '\t' << stat.shedMultiplier << '\t'
                << stat.diagnosedTimestamp << '\t' << stat.hospitalizedTimestamp << '\t'
                << static_cast<unsigned>(stat.infectedCount) << '\t' << static_cast<int>(stat.worstState) << '\t'
                << stat.infectedOccupancy << '\t' << stat.infectiousPresent << '\t' << stat.infectorShed << '\t'
                << stat.infectorInfectedTimestamp << '\t' << stat.symptomOnsetTimestamp << '\t' << stat.quarantinedTimestamp << '\t'
                << stat.quarantinedUntilTimestamp << '\n';
        }
    }

    void infectionsAtLocations(Timehandler& simTime, unsigned timeStep, uint8_t variant) {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        ensureInitialShedding();
        applyPeakWindow(simTime);
        accumulateProfile(simTime);
        if (realThis->locs->eventRate > 0.0) {
            unsigned day = simTime.getTimestamp() / simTime.getStepsPerDay();
            if (!eventLogPath.empty() && eventLogDay >= 0 && static_cast<int>(day) != eventLogDay) flushEventLog();
            realThis->locs->updateDailyEvents(day);
            if (!eventLogPath.empty() && static_cast<int>(day) != eventLogDay) {
                if (eventPresentMax.size() == realThis->locs->eventBoost.size()) {
                    thrust::fill(eventPresentMax.begin(), eventPresentMax.end(), 0u);
                    thrust::fill(eventInfectiousMax.begin(), eventInfectiousMax.end(), 0u);
                }
                eventLogDay = static_cast<int>(day);
            }
        }
        thrust::device_vector<unsigned>& locationListOffsets =
            realThis->locs->locationListOffsets;// offsets into locationAgentList and locationIdsOfAgents
        thrust::device_vector<unsigned>& locationAgentList = realThis->locs->locationAgentList;
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;

        auto& ppstates = realThis->agents->PPValues;
        auto& infectiousness = realThis->locs->infectiousness;

        infectionRatios.resize(locationListOffsets.size() - 1);
        fullInfectedCounts.resize(locationListOffsets.size() - 1);
        thrust::fill(infectionRatios.begin(), infectionRatios.end(), 0.0);
        thrust::fill(fullInfectedCounts.begin(), fullInfectedCounts.end(), 0.0f);

        bool recordNewInfections = flagInfectionAtLocations || !infectionTracePath.empty();
        if (recordNewInfections) {
            if (infectionFlagAtLocations.size() == 0) infectionFlagAtLocations.resize(infectiousness.size());
            thrust::fill(infectionFlagAtLocations.begin(), infectionFlagAtLocations.end(), false);
            if (newlyInfectedAgents.size() == 0) newlyInfectedAgents.resize(ppstates.size());
            thrust::fill(newlyInfectedAgents.begin(), newlyInfectedAgents.end(), 0u);
        }

        //
        // Step 1 - Count up infectious people - those who are Infectious
        //
        Util::reduce_by_location(locationListOffsets,
            locationAgentList,
            fullInfectedCounts,
            ppstates,
            agentLocations,
            realThis->locs->locationIdsOfAgents,
            [variant] HD(const typename SimulationType::PPState_t& ppstate) -> float { return ppstate.isInfectious(variant); });

        dumpToFileStep1(locationListOffsets, locationAgentList, ppstates, fullInfectedCounts, agentLocations, simTime, variant);
        if (!eventLogPath.empty() && realThis->locs->eventRate > 0.0) noteEventExposure();

        //
        // Step 2 - calculate infection ratios, based on density of infected people
        //
        initialiseVolumeModel();
        trackOccupancy(simTime);
        float tmpK = this->k;
        const float* volumePtr = volumeActive ? thrust::raw_pointer_cast(volumeDenominator.data()) : nullptr;
        const bool volumeFill = volumeFillOne;
        float densityPower = static_cast<float>(densityExponent);
        float seasonality = seasonalityMultiplier(simTime);
        //if (simTime.getStepsUntilMidnight()==1) printf("%g\n", seasonality);
        float* eventBoostPtr = thrust::raw_pointer_cast(realThis->locs->eventBoost.data());
        bool hasEventBoost = realThis->locs->eventBoost.size() == infectiousness.size();
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.begin(),
                              locationListOffsets.begin(),
                              locationListOffsets.begin() + 1,
                              infectiousness.begin(),
                              thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(
                fullInfectedCounts.end(), locationListOffsets.end() - 1, locationListOffsets.end(), infectiousness.end(),
                thrust::make_counting_iterator<unsigned>(0) + fullInfectedCounts.size())),
            infectionRatios.begin(),
            [=] HD(thrust::tuple<float&, unsigned&, unsigned&, float&, unsigned> tuple) {
                float numInfectedAgentsPresent = thrust::get<0>(tuple);
                unsigned offset0 = thrust::get<1>(tuple);
                unsigned offset1 = thrust::get<2>(tuple);
                unsigned num_agents = offset1 - offset0;
                if (numInfectedAgentsPresent == 0.0f || num_agents == 0) { return 0.0f; }
                float densityOfInfected = densityPower == 1.0f
                    ? numInfectedAgentsPresent / num_agents
                    : numInfectedAgentsPresent / powf(static_cast<float>(num_agents), densityPower);
                if (volumePtr != nullptr) {
                    float volume = volumePtr[thrust::get<4>(tuple)];
                    if (volume > 0.0f) {
                        densityOfInfected = volumeFill ? numInfectedAgentsPresent / (num_agents * volume)
                                                       : numInfectedAgentsPresent / volume;
                    }
                }
                float boost = hasEventBoost ? eventBoostPtr[thrust::get<4>(tuple)] : 1.0f;
                /*float y =
                    1.0
                    / (1.0
                        + parTMP.v
                              * std::exp(-parTMP.s * 2 * (densityOfInfected - parTMP.h - 0.5)));
                y = parTMP.a * y + parTMP.b;
                y *= thrust::get<3>(tuple);// Weighted by infectiousness
                return y / (60.0 * 24.0 / static_cast<float>(timeStep));*/
                return 1.0f - expf(-tmpK * densityOfInfected * thrust::get<3>(tuple) * boost * seasonality * static_cast<float>(timeStep));
            });

        //
        // Step 3 - randomly infect susceptible people
        //

        LocationsList<SimulationType>::infectAgents(infectionRatios,
            agentLocations,
            infectionFlagAtLocations,
            newlyInfectedAgents,
            recordNewInfections,
            simTime,
            variant,
            sheddingShape,
            getTraitInfectiousnessPtr(),
            traitSusceptibility.empty() ? nullptr : thrust::raw_pointer_cast(traitSusceptibility.data()),
            peakMode);
        if (!infectionTracePath.empty()) attributeInfectors(variant);
        if (flagInfectionAtLocations) {
            thrust::device_vector<unsigned> fullInfectedCounts2(fullInfectedCounts.size(), 0);
            Util::reduce_by_location(locationListOffsets,
                locationAgentList,
                fullInfectedCounts2,
                ppstates,
                agentLocations,
                realThis->locs->locationIdsOfAgents,
                [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned {
                    return unsigned(ppstate.isInfectious() > 0);
                });
            thrust::device_vector<unsigned> numberOfNewInfectionsAtLocations(fullInfectedCounts.size(), 0);
            Util::reduce_by_location(locationListOffsets,
                locationAgentList,
                numberOfNewInfectionsAtLocations,
                newlyInfectedAgents,
                agentLocations,
                realThis->locs->locationIdsOfAgents,
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
