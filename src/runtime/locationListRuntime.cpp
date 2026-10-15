#include "locationList.h"

#include "simulation.h"
#include "configTypes.h"
#include <sstream>

template<typename SimulationType>
void LocationsList<SimulationType>::initLocationTypes(const parser::LocationTypes& inputData) {
    for (auto& type : inputData.types) {
        generalLocationTypes.emplace(std::make_pair(type.ID, std::move(type.name)));
    }
    unsigned cemeteryTypeID = generalLocationTypes.rbegin()->first + 1;
    generalLocationTypes.emplace(std::make_pair(cemeteryTypeID, "cemetery"));
}

template<typename SimulationType>
void LocationsList<SimulationType>::initializeArgs(const cxxopts::ParseResult& result) {
    try {
        tracked = result["trace"].as<unsigned>();
    } catch (std::exception&) {
        tracked = std::numeric_limits<unsigned>::max();
    }
    try {
        std::string profile = result["venueProfile"].as<std::string>();
        if (profile == "covid") venueProfile = 1;
        else if (profile == "caregiving") venueProfile = 2;
        else venueProfile = 0;
        venueStrength = result["venueStrength"].as<double>();
        communityScale = result["communityScale"].as<double>();
        eventRate = result["eventRate"].as<double>();
        eventFactor = result["eventFactor"].as<double>();
        homeVenueFactor = result["homeVenueFactor"].as<double>();
        eventMaxCapacity = result["eventMaxCapacity"].as<double>();
        eventTypeAllowed.assign(64, 0);
        std::stringstream types(result["eventTypes"].as<std::string>());
        std::string item;
        while (std::getline(types, item, ',')) {
            if (item.empty()) continue;
            unsigned type = static_cast<unsigned>(std::stoul(item));
            if (type < eventTypeAllowed.size()) eventTypeAllowed[type] = 1;
        }
    } catch (std::exception&) {
        venueProfile = 0;
        venueStrength = 1.0;
        communityScale = 1.0;
        eventRate = 0.0;
        eventFactor = 20.0;
        homeVenueFactor = -1.0;
        eventMaxCapacity = 0.0;
        eventTypeAllowed.assign(64, 0);
        eventTypeAllowed[5] = eventTypeAllowed[6] = eventTypeAllowed[8] = eventTypeAllowed[9] = 1;
    }
}

namespace {
float covidVenueFactor(unsigned type) {
    switch (type) {
        case 2: return 1.0f;
        case 3: return 1.4f;
        case 33: return 1.6f;
        case 4:
        case 13: return 1.1f;
        case 5: return 4.0f;
        case 6: return 3.0f;
        case 7: return 0.8f;
        case 8: return 2.5f;
        case 9: return 3.0f;
        case 10: return 0.3f;
        case 11: return 2.0f;
        case 12: return 1.8f;
        case 14: return 1.3f;
        case 22: return 2.2f;
        case 44: return 1.5f;
        default: return 1.0f;
    }
}

float caregivingVenueFactor(unsigned type) {
    switch (type) {
        case 2: return 2.5f;
        case 9: return 3.0f;
        case 11: return 2.0f;
        case 12: return 4.0f;
        case 14: return 2.0f;
        case 22: return 3.0f;
        default: return 0.35f;
    }
}
}

template<typename SimulationType>
std::pair<unsigned, std::map<std::string, unsigned>> LocationsList<SimulationType>::initLocations(
    const parser::Locations& inputData,
    const parser::LocationTypes& locTypes) {
    std::map<std::string, unsigned> IDMapping{};

    thrust::host_vector<TypeOfLocation> locType_h;
    thrust::host_vector<PositionType> position_h;
    thrust::host_vector<float> infectiousness_h;
    thrust::host_vector<unsigned> areas_h;
    thrust::host_vector<bool> states_h;
    thrust::host_vector<unsigned> capacity_h;
    thrust::host_vector<uint8_t> essential_h;
    thrust::host_vector<unsigned> quarantineUntil_h;
    auto s = inputData.places.size() + 1;
    locType_h.reserve(s);
    position_h.reserve(s);
    infectiousness_h.reserve(s);
    areas_h.reserve(s);
    states_h.reserve(s);
    capacity_h.reserve(s);
    essential_h.reserve(s);
    quarantineUntil_h.reserve(s);
    thrust::host_vector<unsigned> schools_h;
    thrust::host_vector<std::string> schoolIDs_h;
    thrust::host_vector<unsigned> classrooms_h;
    thrust::host_vector<std::string> classroomsIDs_h;
    thrust::host_vector<unsigned> classroomOffsets_h;

    reserve(s);
    unsigned idx = 0;
    for (unsigned i = 0; i < inputData.places.size(); i++) {
        const auto& loc = inputData.places[i];
        auto it = IDMapping.find(loc.ID);
        if (it != IDMapping.end() && loc.type != locType_h[it->second]) {
            printf(
                "Location with ID %s already exists with mismatching type %d and %d\n",
                loc.ID.c_str(),
                loc.type,
                locType_h[it->second]);
        }
        IDMapping.emplace(loc.ID, idx);
        locType_h.push_back(loc.type);
        position_h.push_back(PositionType{loc.coordinates[0], loc.coordinates[1]});
        infectiousness_h.push_back(loc.infectious);
        capacity_h.push_back(loc.capacity);
        essential_h.push_back(loc.essential);
        areas_h.push_back(loc.area);
        quarantineUntil_h.push_back(0);
        if (loc.type == locTypes.classroom) {
            classrooms_h.push_back(idx);
            classroomsIDs_h.push_back(loc.ID);
        }
        if (loc.type == locTypes.school) {
            schools_h.push_back(idx);
            schoolIDs_h.push_back(loc.ID);
        }
        std::string tmp = loc.state;
        std::for_each(tmp.begin(), tmp.end(), [](char c) { return std::toupper(c); });
        if (tmp == "ON" || tmp == "OPEN") {
            states_h.push_back(true);
        } else if (tmp == "OFF" || tmp == "CLOSED") {
            states_h.push_back(false);
        } else {
            throw IOLocations::WrongState(loc.state);
        }
        idx++;
    }

    locType_h.push_back(generalLocationTypes.rbegin()->first);
    position_h.push_back(PositionType{0, 0});
    infectiousness_h.push_back(0.0f);
    areas_h.push_back(std::numeric_limits<unsigned>::max());
    states_h.push_back(true);
    capacity_h.push_back(std::numeric_limits<unsigned>::max());
    essential_h.push_back(1);

    thrust::host_vector<unsigned> schoolIdForClassroom(classrooms_h.size());
    for (unsigned i = 0; i < classrooms_h.size(); i++) {
        const std::string& s = classroomsIDs_h[i];
        size_t pos = s.find("_");
        if (pos != std::string::npos) {
            std::string schoolid = s.substr(pos + 1);
            auto it = IDMapping.find(schoolid);
            if (it != IDMapping.end()) {
                schoolIdForClassroom[i] = it->second;
            } else {
                throw CustomErrors(
                    "classroom id does not have class_school structure: " + s + " school ID not found " + schoolid);
            }
        } else {
            throw CustomErrors("classroom id does not have class_school structure " + s);
        }
    }
    thrust::stable_sort_by_key(schoolIdForClassroom.begin(), schoolIdForClassroom.end(), classrooms_h.begin());
    for (unsigned i = 0; i < schools_h.size(); i++) {
        auto it = thrust::find(schoolIdForClassroom.begin(), schoolIdForClassroom.end(), schools_h[i]);
        if (it != schoolIdForClassroom.end()) {
            classroomOffsets_h.push_back(std::distance(schoolIdForClassroom.begin(), it));
        } else {
            if (classroomOffsets_h.size() == 0) {
                classroomOffsets_h.push_back(0);
            }
            classroomOffsets_h.push_back(classroomOffsets_h[classroomOffsets_h.size() - 1]);
        }
    }
    classroomOffsets_h.push_back(schoolIdForClassroom.size());
    schools = schools_h;
    classrooms = classrooms_h;
    classroomOffsets = classroomOffsets_h;
    locType = locType_h;
    position = position_h;
    infectiousness = infectiousness_h;
    areas = areas_h;
    states = states_h;
    capacity = capacity_h;
    essential = essential_h;
    quarantineUntil = quarantineUntil_h;

    closedUntil.resize(capacity.size());
    thrust::fill(closedUntil.begin(), closedUntil.end(), 0);

    return std::make_pair(locType.size() - 1, IDMapping);
}

template<typename SimulationType>
void LocationsList<SimulationType>::applyVenueAmplification() {
    // homeVenueFactor is applied only inside this block. The default (-1) leaves the
    // profile-off, community-scale-1 path unchanged, which is what older runs used.
    if (venueProfile != 0 || communityScale != 1.0 || homeVenueFactor >= 0.0) {
        thrust::host_vector<float> infectiousnessHost = infectiousness;
        thrust::host_vector<TypeOfLocation> typesHost = locType;
        for (unsigned i = 0; i < infectiousnessHost.size(); ++i) {
            if (!(infectiousnessHost[i] > 0.0f)) continue;
            unsigned type = typesHost[i];
            float base = 1.0f;
            if (venueProfile == 1) base = covidVenueFactor(type);
            else if (venueProfile == 2) base = caregivingVenueFactor(type);
            if (homeVenueFactor >= 0.0 && type == 2) base = static_cast<float>(homeVenueFactor);
            float factor = 1.0f + static_cast<float>(venueStrength) * (base - 1.0f);
            bool community = !(type == 1 || type == 2 || type == 12 || type == 22);
            if (community) factor *= static_cast<float>(communityScale);
            if (factor < 0.0f) factor = 0.0f;
            infectiousnessHost[i] *= factor;
        }
        infectiousness = infectiousnessHost;
    }
    eventBoost.resize(infectiousness.size());
    thrust::fill(eventBoost.begin(), eventBoost.end(), 1.0f);
    lastEventDay = -1;
}

template<typename SimulationType>
void LocationsList<SimulationType>::updateDailyEvents(unsigned day) {
    if (!(eventRate > 0.0) || lastEventDay == static_cast<int>(day)) return;
    lastEventDay = static_cast<int>(day);
    bool anyEligible = false;
    for (char allowed : eventTypeAllowed) anyEligible = anyEligible || allowed;
    if (!anyEligible) {
        eventTypeAllowed.assign(64, 0);
        eventTypeAllowed[5] = eventTypeAllowed[6] = eventTypeAllowed[8] = eventTypeAllowed[9] = 1;
    }
    if (eventEligible.size() != locType.size()) {
        thrust::host_vector<TypeOfLocation> typesHost = locType;
        thrust::host_vector<unsigned> capacityHost;
        if (eventMaxCapacity > 0.0) capacityHost = capacity;
        thrust::host_vector<uint8_t> eligibleHost(typesHost.size(), 0);
        for (unsigned i = 0; i < typesHost.size(); ++i) {
            unsigned type = typesHost[i];
            bool typeOk = type < eventTypeAllowed.size() && eventTypeAllowed[type];
            bool capOk = !(eventMaxCapacity > 0.0) || (i < capacityHost.size() && static_cast<double>(capacityHost[i]) <= eventMaxCapacity);
            eligibleHost[i] = (typeOk && capOk) ? 1 : 0;
        }
        eventEligible = eligibleHost;
    }
    uint8_t* eligible = thrust::raw_pointer_cast(eventEligible.data());
    float* boost = thrust::raw_pointer_cast(eventBoost.data());
    double rate = eventRate;
    double factor = eventFactor;
    unsigned n = static_cast<unsigned>(locType.size());
    thrust::for_each(thrust::make_counting_iterator<unsigned>(0),
        thrust::make_counting_iterator<unsigned>(n),
        [eligible, boost, rate, factor] HD(unsigned i) {
            // Same draw order as the previous hardcoded types 5, 6, 8 and 9 when those are the eligible set.
            if (eligible[i] && RandomGenerator::randomUnit() < rate) boost[i] = static_cast<float>(factor);
            else boost[i] = 1.0f;
        });
}

template<typename SimulationType>
void LocationsList<SimulationType>::initialize() {
    auto agents = SimulationType::AgentListType::getInstance();
    locationAgentList.resize(agents->location.size());
    locationIdsOfAgents.resize(agents->location.size());
    locationListOffsets.resize(position.size() + 1);
    Util::updatePerLocationAgentLists(agents->location, locationIdsOfAgents, locationAgentList, locationListOffsets);
}

template<typename SimulationType>
const std::vector<unsigned>& LocationsList<SimulationType>::refreshAndGetStatistic() {
    return globalStats.refreshandGetAfterMidnight();
}

template class LocationsList<config::Simulation_t>;
