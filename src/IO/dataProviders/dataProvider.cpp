#include "dataProvider.h"
#include "JSONDecoder.h"
#include <set>
#include <algorithm>
#include "smallTools.h"
#include <list>

void DataProvider::readParameters(const std::string& fileName) {
    parameters = DECODE_JSON_FILE(fileName, decltype(parameters));
}

void DataProvider::readClosureRules(const std::string& fileName) {
    rules = DECODE_JSON_FILE(fileName, decltype(rules));
}

std::map<ProgressionType, std::string> DataProvider::readProgressionConfig(
    const std::string& fileName) {
    progressionConfig = DECODE_JSON_FILE(fileName, parser::ProgressionDirectory);
    std::map<ProgressionType, std::string> progressions;
    auto path = fileName.substr(0, fileName.find_last_of(separator()));
    for (const auto& f : progressionConfig.transitionMatrices) {
        progressions.emplace(std::make_pair(f, path + separator() + f.fileName));
    }
    return progressions;
}

void DataProvider::readProgressionMatrices(const std::string& fileName) {
    auto rawProgressions = readProgressionConfig(fileName);
    std::list<std::pair<unsigned, unsigned>> ageInters;
    for (const auto& kv : rawProgressions) {
        bool inserted = false;
        for (auto it = ageInters.begin(); it != ageInters.end(); ++it) {
            if (kv.first.ageEnd < it->first) {
                ageInters.insert(it, std::make_pair(kv.first.ageBegin, kv.first.ageEnd));
                inserted = true;
                break;
            } else if ((it->first == kv.first.ageBegin) && (it->second == kv.first.ageEnd)) {
                inserted = true;
                break;
            }
        }
        if (!inserted) { ageInters.push_back(std::make_pair(kv.first.ageBegin, kv.first.ageEnd)); }
    }
    auto numberCond = parameters.preCondition.size();
    for (const auto& kv : rawProgressions) {
        auto currentAgeBegin = kv.first.ageBegin;
        auto currentPreCond = kv.first.preCond;
        auto ageIndex = std::distance(ageInters.begin(),
            std::find_if(ageInters.begin(), ageInters.end(), [currentAgeBegin](const auto& e) {
                return e.first == currentAgeBegin;
            }));

        auto condIndex = std::distance(parameters.preCondition.begin(),
            std::find_if(parameters.preCondition.begin(),
                parameters.preCondition.end(),
                [currentPreCond](const auto& e) { return e.ID == currentPreCond; }));
        unsigned index = ageIndex * numberCond + condIndex;

        progressionDirectory.emplace(std::make_pair(kv.first,
            std::make_pair(DECODE_JSON_FILE(kv.second, parser::TransitionFormat), index)));
    }
}

void DataProvider::readConfigRandom(const std::string& fileName) {
    configRandom = DECODE_JSON_FILE(fileName, decltype(configRandom));
}

void DataProvider::readAgentTypes(const std::string& fileName) {
    agentTypes = DECODE_JSON_FILE(fileName, decltype(agentTypes));
    for (const auto& aType : agentTypes.types) {
        std::set<unsigned> locs{ locationTypes.hospital,
            locationTypes.publicSpace,
            locationTypes.home,
            locationTypes.doctor };//, locationTypes.school, locationTypes.work };
        for (const auto& sch : aType.schedulesUnique) {
            for (const auto& event : sch.schedule) { locs.insert(event.type); }
        }
        aTypeToLocationTypes.emplace(std::piecewise_construct,
            std::forward_as_tuple(aType.ID),
            std::forward_as_tuple(locs.begin(), locs.end()));
    }
}

void DataProvider::readLocationTypes(const std::string& fileName) {
    locationTypes = DECODE_JSON_FILE(fileName, decltype(locationTypes));
}

void DataProvider::readLocations(const std::string& fileName, bool randomAgents) {
    locations = DECODE_JSON_FILE(fileName, decltype(locations));
    if (randomAgents) {
        for (const auto& l : locations.places) { typeToLocationMapping[l.type].push_back(l.ID); }
    }
}

void DataProvider::readAgents(const std::string& fileName) {
    agents = DECODE_JSON_FILE(fileName, decltype(agents));
}

std::pair<std::string, double> DataProvider::calculateSingleRandomState(unsigned age) const {
    auto it = std::find_if(configRandom.stateDistibution.begin(),
        configRandom.stateDistibution.end(),
        [age](const auto& e) { return (e.ageStart <= age) && (age < e.ageEnd); });
    return randomSelectPair(it->distribution.begin());
}


void DataProvider::randomLocations(unsigned N) {
    locations.places.reserve(N);
    auto locTypes = static_cast<unsigned>(locationTypes.types.size());
    for (unsigned i = 0; i < N; ++i) {
        parser::Locations::Place current{};
        current.ID = std::to_string(i);
        current.type = randomSelect(configRandom.locationTypeDistibution.begin());
        typeToLocationMapping[current.type].push_back(current.ID);
        current.coordinates = std::vector<double>{ 0.0, 0.0 };
        current.area = 1;
        current.state = "ON";
        current.capacity = 100;
        current.ageInter = std::vector<int>{ 0, 100 };
        current.infectious = 1.0;
        if (current.type == 4 || current.type == 7 || current.type == 8)
            current.essential = RandomGenerator::randomUnit() < 0.1;
        else if (current.type == 12 || current.type == 14)
            current.essential = 1;
        else 
            current.essential = 0;
        locations.places.emplace_back(std::move(current));
    }
}

void DataProvider::randomAgents(unsigned N) {
    agents.people.reserve(N);
    for (unsigned i = 0; i < N; ++i) {
        parser::Agents::Person current{};
        current.age = RandomGenerator::randomUnsigned(90);
        current.sex = (RandomGenerator::randomUnit() < 0.5) ? "M" : "F";
        current.preCond = randomSelect(configRandom.preCondDistibution.begin());
        auto statediag = calculateSingleRandomState(current.age);
        current.state = statediag.first;
        current.diagnosed = RandomGenerator::randomUnit() < statediag.second;
        current.typeID = randomSelect(configRandom.agentTypeDistribution.begin());
        const auto& requestedLocations = aTypeToLocationTypes[current.typeID];
        current.locations.reserve(requestedLocations.size());
        for (const auto& l : requestedLocations) {
            parser::Agents::Person::Location currentLoc{};
            currentLoc.typeID = l;
            const auto& possibleLocations = typeToLocationMapping[currentLoc.typeID];
            if ((possibleLocations.size() == 0)
                || (RandomGenerator::randomUnit() < configRandom.irregulalLocationChance)) {
                currentLoc.locID =
                    locations.places[RandomGenerator::randomUnsigned(locations.places.size())].ID;
            } else {
                auto r = RandomGenerator::randomUnsigned(possibleLocations.size());
                currentLoc.locID = possibleLocations[r];
            }
            current.locations.push_back(currentLoc);
        }
        agents.people.emplace_back(std::move(current));
    }
}

void DataProvider::randomStates() {
    for (auto& a : agents.people) { 
        auto statediag = calculateSingleRandomState(a.age);
        a.state = statediag.first;
        a.diagnosed = RandomGenerator::randomUnit() < statediag.second;
    }
}

DataProvider::DataProvider(const cxxopts::ParseResult& result) {
    //PROFILE_FUNCTION();
    readParameters(result["parameters"].as<std::string>());
    readProgressionMatrices(result["progression"].as<std::string>());
    readClosureRules(result["closures"].as<std::string>());
    int numberOfAgents = result["numagents"].as<int>();
    int numberOfLocations = result["numlocs"].as<int>();
    if ((numberOfAgents != -1) || (numberOfLocations != -1) || result["randomStates"].as<bool>()) {
        readConfigRandom(result["configRandom"].as<std::string>());
    }
    readLocationTypes(result["locationTypes"].as<std::string>());
    readAgentTypes(result["agentTypes"].as<std::string>());
    if (numberOfLocations == -1) {
        readLocations(result["locations"].as<std::string>(), numberOfAgents == -1);
    } else {
        randomLocations(numberOfLocations);
    }
    if (numberOfAgents == -1) {
        readAgents(result["agents"].as<std::string>());
        if (result["randomStates"].as<bool>()) { randomStates(); }
    } else {
        randomAgents(numberOfAgents);
    }
}

[[nodiscard]] parser::Agents& DataProvider::acquireAgents() { return agents; }
[[nodiscard]] parser::AgentTypes& DataProvider::acquireAgentTypes() { return agentTypes; }

[[nodiscard]] parser::Locations& DataProvider::acquireLocations() { return locations; }

[[nodiscard]] parser::LocationTypes& DataProvider::acquireLocationTypes() { return locationTypes; }

[[nodiscard]] parser::ClosureRules& DataProvider::acquireClosureRules() { return rules; }

[[nodiscard]] parser::Parameters& DataProvider::acquireParameters() { return parameters; }

[[nodiscard]] std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>&
    DataProvider::acquireProgressionMatrices() {
    return progressionDirectory;
}

[[nodiscard]] parser::ProgressionDirectory& DataProvider::acquireProgressionConfig() {
    return progressionConfig;
}

[[nodiscard]] const std::map<unsigned, std::vector<unsigned>>&
    DataProvider::getAgentTypeLocTypes() const {
    return aTypeToLocationTypes;
}
