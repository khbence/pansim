#include "locationList.h"

#include "simulation.h"
#include "configTypes.h"

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
