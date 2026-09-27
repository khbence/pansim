#include "immunization.h"

#include "simulation.h"
#include "configTypes.h"

namespace {

std::vector<float> parseWeeklyRatesFile(const std::string& path) {
    std::ifstream t(path.c_str());
    std::stringstream buffer;
    buffer << t.rdbuf();
    return splitStringFloat(buffer.str(), ',');
}

void parseDailyOrWeeklyRate(const std::string& input, unsigned& dailyRate, std::vector<float>& weeklyRates) {
    char* endptr = nullptr;
    dailyRate = strtoul(input.c_str(), &endptr, 10) / 7;
    if (endptr != input.c_str() + input.length()) {
        weeklyRates = parseWeeklyRatesFile(input);
    }
}

template<class Simulation>
void parseBoosterScheduleConfig(
    const std::string& input,
    unsigned boosterRounds,
    unsigned& dailyBoosters,
    std::vector<std::vector<float>>& boosterPerWeek,
    std::vector<std::vector<float>>& boosterAgePercent) {
    char* endptr = nullptr;
    dailyBoosters = strtoul(input.c_str(), &endptr, 10) / 7;
    if (endptr == input.c_str() + input.length()) {
        return;
    }

    std::ifstream t(input.c_str());
    std::string buffer;
    unsigned localBoosterRounds = 0;
    while (std::getline(t, buffer)) {
        if (buffer.length() == 0 || buffer.at(0) == '#') {
            continue;
        }
        boosterPerWeek[localBoosterRounds] = splitStringFloat(buffer, ',');
        if (std::getline(t, buffer)) {
            boosterAgePercent[localBoosterRounds] = splitStringFloat(buffer, ',');
        }
        localBoosterRounds++;
        if (localBoosterRounds == boosterRounds) {
            break;
        }
    }
    if (boosterRounds != localBoosterRounds) {
        throw CustomErrors("booster rounds size mismatch");
    }
}

void validateVaccinationOrder(const std::vector<int>& vaccinationOrder) {
    if (vaccinationOrder.size() != numberOfCategories) {
        throw CustomErrors("immunizationOrder mush have exactly " + std::to_string(numberOfCategories) + " values");
    }
    for (int i = 0; i < numberOfCategories; i++) {
        if (vaccinationOrder[i] > numberOfCategories) {
            throw CustomErrors("immunizationOrder values have to be  less or equal to " + std::to_string(numberOfCategories));
        }
    }
}

} // namespace

template<class Simulation>
void Immunization<Simulation>::initializeArgs(const cxxopts::ParseResult& result) {
    startAfterDay = result["immunizationStart"].as<unsigned>();
    boosterStartAfterDay = splitStringInt(result["boosterStart"].as<std::string>(), ',');
    boosterRounds = boosterStartAfterDay.size();
    boosterPerWeek.resize(boosterRounds);
    boosterAgePercent.resize(boosterRounds);

    parseDailyOrWeeklyRate(result["immunizationsPerWeek"].as<std::string>(), dailyDoses, vaccPerWeek);
    parseBoosterScheduleConfig<Simulation>(
        result["boostersPerWeek"].as<std::string>(), boosterRounds, dailyBoosters, boosterPerWeek, boosterAgePercent);

    ageGroupSize.resize(boosterRounds);
    ageGroupFrac.resize(boosterRounds);

    try {
        diagnosticLevel = result["diags"].as<unsigned>();
    } catch (std::exception&) {
    }

    vaccinationOrder = splitStringInt(result["immunizationOrder"].as<std::string>(), ',');
    validateVaccinationOrder(vaccinationOrder);

    protectionInfection = splitStringFloat(result["protectionInfection"].as<std::string>(), ',');
    protectionInfectionWaning = splitStringFloat(result["protectionInfectionWaning"].as<std::string>(), ',');
    protectionSymptomatic = splitStringFloat(result["protectionSymptomatic"].as<std::string>(), ',');
    protectionSymptomaticWaning = splitStringFloat(result["protectionSymptomaticWaning"].as<std::string>(), ',');
    protectionHospitalization = splitStringFloat(result["protectionHospitalization"].as<std::string>(), ',');
    protectionHospitalizationWaning = splitStringFloat(result["protectionHospitalizationWaning"].as<std::string>(), ',');
    variantSimilarMultiplier = splitStringFloat(result["variantSimilarMultiplier"].as<std::string>(), ',');
    variantSimilarity = splitStringInt(result["variantSimilarity"].as<std::string>(), ',');
    vaccinationGroupLevel = splitStringFloat(result["vaccinationGroupLevel"].as<std::string>(), ',');
    numVariants = sim->infectiousnessMultiplier.size();
}

template<class Simulation>
void Immunization<Simulation>::initCategories() {
    immunizationRound.resize(sim->agents->PPValues.size(), 0);

    auto* agentMetaDataPtr = thrust::raw_pointer_cast(sim->agents->agentMetaData.data());
    auto* locationOffsetPtr = thrust::raw_pointer_cast(sim->agents->locationOffset.data());
    auto* possibleTypesPtr = thrust::raw_pointer_cast(sim->agents->possibleTypes.data());
    auto* locationTypePtr = thrust::raw_pointer_cast(sim->locs->locType.data());
    auto* possibleLocationsPtr = thrust::raw_pointer_cast(sim->agents->possibleLocations.data());
    auto* essentialPtr = thrust::raw_pointer_cast(sim->locs->essential.data());

    float cat0_lvl = vaccinationGroupLevel[0];
    auto cat_healthworker = [locationOffsetPtr, possibleTypesPtr, possibleLocationsPtr, locationTypePtr, cat0_lvl] HD(
                                unsigned id) -> thrust::pair<bool, float> {
        for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
            if (possibleTypesPtr[idx] == 4 &&
                (locationTypePtr[possibleLocationsPtr[idx]] == 12 || locationTypePtr[possibleLocationsPtr[idx]] == 14)) {
                return thrust::make_pair(true, cat0_lvl);
            }
        }
        return thrust::make_pair(false, 0.0f);
    };

    float cat1_lvl = vaccinationGroupLevel[1];
    auto cat_nursery = [locationOffsetPtr, possibleTypesPtr, locationTypePtr, possibleLocationsPtr, cat1_lvl] HD(
                           unsigned id) -> thrust::pair<bool, float> {
        for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
            if ((possibleTypesPtr[idx] == 4 || possibleTypesPtr[idx] == 2) &&
                locationTypePtr[possibleLocationsPtr[idx]] == 22) {
                return thrust::make_pair(true, cat1_lvl);
            }
        }
        return thrust::make_pair(false, 0.0f);
    };

    float cat2_lvl = vaccinationGroupLevel[2];
    auto cat_elderly_underlying = [agentMetaDataPtr, cat2_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
        if (agentMetaDataPtr[id].getPrecondIdx() > 0 && agentMetaDataPtr[id].getAge() >= 60) {
            return thrust::make_pair(true, cat2_lvl);
        }
        return thrust::make_pair(false, 0.0f);
    };

    float cat3_lvl = vaccinationGroupLevel[3];
    auto cat_elderly = [agentMetaDataPtr, cat3_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
        if (agentMetaDataPtr[id].getAge() >= 60) {
            return thrust::make_pair(true, cat3_lvl);
        }
        return thrust::make_pair(false, 0.0f);
    };

    float cat4_lvl = vaccinationGroupLevel[4];
    auto cat_underlying = [agentMetaDataPtr, cat4_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
        if (agentMetaDataPtr[id].getPrecondIdx() > 0 && agentMetaDataPtr[id].getAge() >= 18 &&
            agentMetaDataPtr[id].getAge() < 60) {
            return thrust::make_pair(true, cat4_lvl);
        }
        return thrust::make_pair(false, 0.0f);
    };

    float cat5_lvl = vaccinationGroupLevel[5];
    auto cat_essential = [locationOffsetPtr, possibleTypesPtr, essentialPtr, possibleLocationsPtr, cat5_lvl] HD(
                             unsigned id) -> thrust::pair<bool, float> {
        for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
            if (possibleTypesPtr[idx] == 4 && essentialPtr[possibleLocationsPtr[idx]] == 1) {
                return thrust::make_pair(true, cat5_lvl);
            }
        }
        return thrust::make_pair(false, 0.0f);
    };

    float cat6_lvl = vaccinationGroupLevel[6];
    auto cat_school = [locationOffsetPtr, possibleTypesPtr, locationTypePtr, possibleLocationsPtr, cat6_lvl] HD(
                          unsigned id) -> thrust::pair<bool, float> {
        for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
            if (possibleTypesPtr[idx] == 4 && locationTypePtr[possibleLocationsPtr[idx]] == 3) {
                return thrust::make_pair(true, cat6_lvl);
            }
        }
        return thrust::make_pair(false, 0.0f);
    };

    float cat7_lvl = vaccinationGroupLevel[7];
    auto cat_adult = [agentMetaDataPtr, cat7_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
        if (agentMetaDataPtr[id].getAge() > 17 && agentMetaDataPtr[id].getAge() < 60) {
            return thrust::make_pair(true, cat7_lvl);
        }
        return thrust::make_pair(false, 0.0f);
    };

    float cat8_lvl = vaccinationGroupLevel[8];
    auto cat_child = [agentMetaDataPtr, cat8_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
        if (agentMetaDataPtr[id].getAge() >= 12 && agentMetaDataPtr[id].getAge() < 18) {
            return thrust::make_pair(true, cat8_lvl);
        }
        return thrust::make_pair(false, 0.0f);
    };

    float cat9_lvl = vaccinationGroupLevel[9];
    auto cat_child5 = [agentMetaDataPtr, cat9_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
        if (agentMetaDataPtr[id].getAge() >= 5 && agentMetaDataPtr[id].getAge() < 12) {
            return thrust::make_pair(true, cat9_lvl);
        }
        return thrust::make_pair(false, 0.0f);
    };

    uint8_t lorder[numberOfCategories];
    for (unsigned i = 0; i < numberOfCategories; i++) {
        lorder[i] = vaccinationOrder[i];
    }

    for (unsigned i = 0; i < numberOfCategories; i++) {
        auto it = std::find(vaccinationOrder.begin(), vaccinationOrder.end(), i + 1);
        while (it != vaccinationOrder.end()) {
            auto it = std::find(vaccinationOrder.begin(), vaccinationOrder.end(), i + 1);
            if (it == vaccinationOrder.end()) {
                break;
            }
            *it = -1 * (*it);
            unsigned groupIdx = std::distance(vaccinationOrder.begin(), it);
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.begin(), thrust::make_counting_iterator(0))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    immunizationRound.end(), thrust::make_counting_iterator((int)immunizationRound.size()))),
                [cat_healthworker,
                    cat_nursery,
                    cat_elderly,
                    cat_underlying,
                    cat_essential,
                    cat_adult,
                    cat_elderly_underlying,
                    cat_school,
                    cat_child,
                    cat_child5,
                    lorder,
                    groupIdx] HD(thrust::tuple<uint8_t&, int> tup) {
                    uint8_t& round = thrust::get<0>(tup);
                    unsigned id = thrust::get<1>(tup);

                    auto ret = cat_healthworker(id);
                    if (groupIdx == 0 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[0] : (uint8_t)-1;
                        return;
                    }

                    ret = cat_nursery(id);
                    if (groupIdx == 1 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[1] : (uint8_t)-1;
                        return;
                    }

                    ret = cat_elderly(id);
                    if (groupIdx == 2 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[2] : (uint8_t)-1;
                        return;
                    }

                    ret = cat_underlying(id);
                    if (groupIdx == 3 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[3] : (uint8_t)-1;
                        return;
                    }

                    ret = cat_essential(id);
                    if (groupIdx == 4 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[4] : (uint8_t)-1;
                        return;
                    }

                    ret = cat_adult(id);
                    if (groupIdx == 5 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[5] : (uint8_t)-1;
                        return;
                    }

                    ret = cat_elderly_underlying(id);
                    if (groupIdx == 6 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[6] : (uint8_t)-1;
                        return;
                    }

                    ret = cat_school(id);
                    if (groupIdx == 7 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[7] : (uint8_t)-1;
                        return;
                    }

                    ret = cat_child(id);
                    if (groupIdx == 8 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[8] : (uint8_t)-1;
                        return;
                    }

                    ret = cat_child5(id);
                    if (groupIdx == 9 && ret.first && round == 0) {
                        round = RandomGenerator::randomUnit() < ret.second ? lorder[9] : (uint8_t)-1;
                        return;
                    }
                });
        }
    }
    for (unsigned i = 0; i < numberOfCategories; i++) {
        vaccinationOrder[i] = -1 * vaccinationOrder[i];
    }
}

template<class Simulation>
void Immunization<Simulation>::initAgeGroups() {
    initializedAgeGroups = 1;
    for (int boosterRound = 0; boosterRound < boosterRounds; boosterRound++) {
        for (int agegroup = 0; agegroup < 10; agegroup++) {
            unsigned low = agegroup * 10;
            unsigned high = agegroup * 10 + 10;
            if (agegroup == 9) {
                high = 200;
            }
            ageGroupSize[boosterRound][agegroup] = thrust::count_if(
                sim->agents->agentMetaData.begin(),
                sim->agents->agentMetaData.end(),
                [low, high] HD(typename Simulation::AgentMeta_t meta) { return meta.getAge() >= low && meta.getAge() < high; });
            ageGroupSize[boosterRound][agegroup] =
                (float)ageGroupSize[boosterRound][agegroup] * boosterAgePercent[boosterRound][agegroup];
        }
        unsigned sum = std::accumulate(ageGroupSize[boosterRound].begin(), ageGroupSize[boosterRound].end(), (unsigned)0);
        for (int agegroup = 0; agegroup < 10; agegroup++) {
            ageGroupFrac[boosterRound][agegroup] = (float)ageGroupSize[boosterRound][agegroup] / (float)sum;
        }
    }
}

template class Immunization<config::Simulation_t>;
