#pragma once
#include "agentsFormat.h"
#include "agentTypesFormat.h"
#include "locationsFormat.h"
#include "locationTypesFormat.h"
#include "parametersFormat.h"
#include "progressionConfigFormat.h"
#include "closuresFormat.h"
#include "progressionMatrixFormat.h"
#include "configRandomFormat.h"
#include "progressionType.h"
#include "cxxopts.hpp"

#include "randomGenerator.h"
#include <map>
#include <vector>
#include "timing.h"
#include <utility>

class DataProvider {
    parser::Agents agents;
    parser::AgentTypes agentTypes;
    parser::Locations locations;
    parser::LocationTypes locationTypes;
    parser::Parameters parameters;
    parser::ClosureRules rules;
    std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>
        progressionDirectory;
    parser::ProgressionDirectory progressionConfig;

    // only for random generations and checking
    parser::ConfigRandom configRandom;
    std::map<unsigned, std::vector<unsigned>> aTypeToLocationTypes;
    std::map<unsigned, std::vector<std::string>> typeToLocationMapping;
    std::map<std::string, parser::ConfigRandom::IrregularChances::Detail> typesIrregularChancesMap;

    void readParameters(const std::string& fileName);
    std::map<ProgressionType, std::string> readProgressionConfig(const std::string& fileName);
    void readProgressionMatrices(const std::string& fileName);
    void readConfigRandom(const std::string& fileName);
    void readLocationTypes(const std::string& fileName);
    void readLocations(const std::string& fileName, bool randomAgents);
    void readAgentTypes(const std::string& fileName);
    void readAgents(const std::string& fileName);
    void readClosureRules(const std::string& fileName);

    template<typename Iter>
    [[nodiscard]] auto randomSelect(Iter it) const {
        double r = RandomGenerator::randomUnit();
        long double preSum = it->chance;
        while (preSum < r) {
            ++it;
            preSum += it->chance;
        }
        return it->value;
    }

    template<typename Iter>
    [[nodiscard]] auto randomSelectPair(Iter it) const {
        double r = RandomGenerator::randomUnit();
        long double preSum = it->chance;
        while (preSum < r) {
            ++it;
            preSum += it->chance;
        }
        return std::make_pair(it->value, it->diagnosedChance);
    }

    [[nodiscard]] std::pair<std::string, double> calculateSingleRandomState(unsigned age) const;

    void randomLocations(unsigned N);
    void randomAgents(unsigned N);
    void randomStates();

public:
    explicit DataProvider(const cxxopts::ParseResult& result);

    [[nodiscard]] parser::Agents& acquireAgents();
    [[nodiscard]] parser::AgentTypes& acquireAgentTypes();
    [[nodiscard]] parser::Locations& acquireLocations();
    [[nodiscard]] parser::LocationTypes& acquireLocationTypes();
    [[nodiscard]] parser::Parameters& acquireParameters();
    [[nodiscard]] parser::ClosureRules& acquireClosureRules();
    [[nodiscard]] std::
        map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>&
        acquireProgressionMatrices();
    [[nodiscard]] parser::ProgressionDirectory& acquireProgressionConfig();

    [[nodiscard]] const std::map<unsigned, std::vector<unsigned>>& getAgentTypeLocTypes() const;
};