#pragma once
#include <vector>
#include "datatypes.h"
#include <string>
#include "agentType.h"
#include <map>
#include "parametersFormat.h"
#include "agentTypesFormat.h"
#include "customExceptions.h"
#include "globalStates.h"
#include "timeHandler.h"
#include <iterator>
#include "agentsFormat.h"
#include "agentMeta.h"
#include "agentStats.h"
#include "agentStatOutput.h"
#include "progressionMatrixFormat.h"
#include "dataProvider.h"
#include "progressionType.h"
#include "smallTools.h"

template<class Simulation>
class Immunization {
    Simulation* sim;
    thrust::device_vector<uint8_t> immunizationRound;
    unsigned currentCategory = 0;
#define numberOfCategories 10
    std::vector<int> vaccinationOrder;
    std::vector<float> vaccinationGroupLevel;
    std::vector<float> immunizationEfficiencyInfection;
    std::vector<float> immunizationEfficiencyProgression;
    std::vector<float> acquiredMultiplier;
    std::vector<float> vaccPerWeek;
    std::vector<float> boosterPerWeek;
    unsigned startAfterDay = 0;
    unsigned boosterStartAfterDay = 0;
    unsigned dailyDoses = 0;
    unsigned dailyBoosters = 0;
    unsigned diagnosticLevel = 0;
    unsigned numVariants = 1;
#define pct75 0
    unsigned numberOfVaccinesToday(Timehandler& simTime) {
        #if pct75==1
        /* kelet+nyugat 75%-ra */
        // float availPerWeek[] = {1083, 919, 395, 1599, 796, 1038, 1726, 4630, 5703,6052,4474, 5897, 7951, 9656, 6495, 5994, 8424, 8424, 8424, 8424, 8424, 8424, 8424, 8424, 8424, 8424, 8424, 8424, 8424, 8424};
        #else
        // /* kelet+nyugat */ float availPerWeek[] = {189, 941, 1074, 465, 1410, 1183, 897, 2028, 4266, 5581, 6208, 4483, 6252, 7515, 9617, 7656, 6313, 8601, 4124, 5721, 7454, 2413, 2267, 1187, 1177, 1209, 754, 616, 547, 535, 533, 452, 498, 445, 590, 1364, 650, 369, 340, 325, 268, 259, 280, 277, 321, 448, 472, 1280, 1075, 1075, 1075};
        #endif
        //float availPerWeek[] = {189, 941, 1073, 465, 1410, 1182, 896, 2027, 4264, 5578, 6206, 4481, 6249, 7512, 9613, 7653, 6310, 8597, 4122, 5719, 7451, 2412, 2266, 1581, 1177, 1209, 753, 616, 546, 534, 532, 452, 498, 543, 590, 1364};
        /* csak nyugat */ // float availPerWeek[] = {1083, 919, 395, 1599, 796, 994, 1435,1734, 2055, 3572, 2671,3937 ,5097, 5935, 3564,5283,2447,1705,2778,5727,2686,2686, 2686, 2686, 2686, 2686, 2686, 2686, 2686, 2686, 2686, 2686, 2686, 2686, 2686, 2686};
        if (simTime.getTimestamp() / (24 * 60 / simTime.getTimeStep()) >= startAfterDay) {
            if (vaccPerWeek.size() == 0) return dailyDoses;
            unsigned day = simTime.getTimestamp()/(24*60/simTime.getTimeStep())-startAfterDay;
            unsigned week = day/7;
            return vaccPerWeek[week>=vaccPerWeek.size()?vaccPerWeek.size()-1:week]/7.0;
        } else
            return 0;
    }

    unsigned numberOfBoostersToday(Timehandler& simTime) {
        // float availPerWeek[] = {955, 1433, 1190, 1911, 2095, 1966, 1856, 1819, 1672, 1746, 1856, 1984, 1856, 3234, 4263, 5707, 10580, 5050, 4128, 4128, 4128, 4128, 4128, 0, 0};
        if (simTime.getTimestamp() / (24 * 60 / simTime.getTimeStep()) >= boosterStartAfterDay) {
            if (boosterPerWeek.size() == 0) return dailyBoosters;
            unsigned day = simTime.getTimestamp()/(24*60/simTime.getTimeStep())-boosterStartAfterDay;
            unsigned week = day/7;
            return boosterPerWeek[week>=boosterPerWeek.size()?boosterPerWeek.size()-1:week]/7.0;
        } else
            return 0;
    }

public:
    unsigned immunizedToday = 0;

    Immunization(Simulation* s) { this->sim = s; }
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("immunizationStart",
            "number of days into simulation when immunization starts",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(96))))
            ("boosterStart",
            "number of days into simulation when booster immunizations starts",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(315))))("immunizationsPerWeek",
            "number of immunizations per week - number or file with comma separated values for each week",
            cxxopts::value<std::string>()->default_value("inputConfigFiles/vaccPerWeek.txt"))
            ("boostersPerWeek",
            "number of boosters per week - number or file with comma separated values for each week",
            cxxopts::value<std::string>()->default_value("inputConfigFiles/boosterPerWeek.txt"))
            ("immunizationOrder",
            "Order of immunization (starting at 1, 0 to skip) for agents in different categories health workers, nursery home "
            "worker/resident, 60+, 18-60 with underlying condition, essential worker, 18+, 60+underlying, school teachers, "
            "12-18 children, 5-12 children",
            cxxopts::value<std::string>()->default_value("1,2,3,4,5,6,0,0,7,0"))
            ("vaccinationGroupLevel",
            "Immunization level in different groups in the order listed in immunizationOrder's description",
            cxxopts::value<std::string>()->default_value("0.9,0.85,0.9,0.82,0.8,0.75,0.8,0.67,0.4,0.2")) //75%: "0.9,0.85,0.9,0.89,0.95,0.85,0.8,0.88,0.6,0.4"
            ("immunizationEfficiencyInfection",
            "Efficiency of immunization against infection after 12 days, 28 days, and after booster (pairs of comma-separated values for different strains)",
            cxxopts::value<std::string>()->default_value("0.52,0.96,0.99,0.2,0.82,0.95,0.09,0.71,0.91"))
            ("immunizationEfficiencyProgression",
            "Efficiency of immunization in mitigating disease progression after 12 days, 28 days, and after booster (pairs of comma-separated values for different strains)",
            cxxopts::value<std::string>()->default_value("1.0,1.0,1.0,0.4,0.22,0.1,0.4,0.22,0.1"))
            ("acquiredMultiplier",
            "susceptibility vs. reinfection and progression weight with acquired immunity ",
            cxxopts::value<std::string>()->default_value("0.9,0.22,0.8,0.22,0.8,0.22"));
    }

    void initializeArgs(const cxxopts::ParseResult& result) {
        startAfterDay = result["immunizationStart"].as<unsigned>();
        boosterStartAfterDay = result["boosterStart"].as<unsigned>();
        std::string immStr = result["immunizationsPerWeek"].as<std::string>();
        char *endptr = nullptr;
        dailyDoses = strtoul(immStr.c_str(), &endptr, 10)/7;
        if (endptr != immStr.c_str()+immStr.length()) {
            std::ifstream t(immStr.c_str());
            std::stringstream buffer;
            buffer << t.rdbuf();
            vaccPerWeek = splitStringFloat(buffer.str(), ',');
        }
        std::string bstrStr = result["boostersPerWeek"].as<std::string>();
        dailyBoosters = strtoul(bstrStr.c_str(), &endptr, 10)/7;
        if (endptr != bstrStr.c_str()+bstrStr.length()) {
            std::ifstream t(bstrStr.c_str());
            std::stringstream buffer;
            buffer << t.rdbuf();
            boosterPerWeek = splitStringFloat(buffer.str(), ',');
        }
        
        try {
            diagnosticLevel = result["diags"].as<unsigned>();
        } catch (std::exception& e) {}

        acquiredMultiplier = splitStringFloat(result["acquiredMultiplier"].as<std::string>(),',');

        std::string orderString = result["immunizationOrder"].as<std::string>();
        vaccinationOrder = splitStringInt(orderString, ',');
        
        if (vaccinationOrder.size() != numberOfCategories)
            throw CustomErrors("immunizationOrder mush have exactly " + std::to_string(numberOfCategories) + " values");
        for (int i = 0; i < numberOfCategories; i++)
            if (vaccinationOrder[i] > numberOfCategories)
                throw CustomErrors(
                    "immunizationOrder values have to be  less or equal to " + std::to_string(numberOfCategories));
        
        immunizationEfficiencyInfection = splitStringFloat(result["immunizationEfficiencyInfection"].as<std::string>(), ',');
        immunizationEfficiencyProgression = splitStringFloat(result["immunizationEfficiencyProgression"].as<std::string>(), ',');
        vaccinationGroupLevel = splitStringFloat(result["vaccinationGroupLevel"].as<std::string>(), ',');
        numVariants = sim ->infectiousnessMultiplier.size();
        if (immunizationEfficiencyInfection.size() < 3 * numVariants ||
            immunizationEfficiencyProgression.size() < 3 * numVariants) {
                throw CustomErrors(
                    "immunizationEfficiency parameters have to be at least of size 3x the number of strains");
            }
    }

    void initCategories() {
        immunizationRound.resize(sim->agents->PPValues.size(), 0);

        auto* agentMetaDataPtr = thrust::raw_pointer_cast(sim->agents->agentMetaData.data());
        auto* locationOffsetPtr = thrust::raw_pointer_cast(sim->agents->locationOffset.data());
        auto* possibleTypesPtr = thrust::raw_pointer_cast(sim->agents->possibleTypes.data());
        auto* locationTypePtr = thrust::raw_pointer_cast(sim->locs->locType.data());
        auto* possibleLocationsPtr = thrust::raw_pointer_cast(sim->agents->possibleLocations.data());
        auto* essentialPtr = thrust::raw_pointer_cast(sim->locs->essential.data());


        // Figure out which category agents belong to, and determine if agent is willing to be vaccinated

        // Category health worker
        float cat0_lvl = vaccinationGroupLevel[0];
        auto cat_healthworker = [locationOffsetPtr, possibleTypesPtr, possibleLocationsPtr, locationTypePtr, cat0_lvl] HD(
                                    unsigned id) -> thrust::pair<bool, float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
                if (possibleTypesPtr[idx] == 4
                    && (locationTypePtr[possibleLocationsPtr[idx]] == 12 || locationTypePtr[possibleLocationsPtr[idx]] == 14))
                    return thrust::make_pair(true, cat0_lvl);
            }
            return thrust::make_pair(false, 0.0f);
        };

        // Category nursery home workers & residents
        float cat1_lvl = vaccinationGroupLevel[1];
        auto cat_nursery = [locationOffsetPtr, possibleTypesPtr, locationTypePtr, possibleLocationsPtr, cat1_lvl] HD(
                               unsigned id) -> thrust::pair<bool, float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
                if ((possibleTypesPtr[idx] == 4 || possibleTypesPtr[idx] == 2)
                    && locationTypePtr[possibleLocationsPtr[idx]] == 22)
                    return thrust::make_pair(true, cat1_lvl);
            }
            return thrust::make_pair(false, 0.0f);
        };

        // Category elderly with underlying condition
        float cat2_lvl = vaccinationGroupLevel[2];
        auto cat_elderly_underlying = [agentMetaDataPtr, cat2_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
            if (agentMetaDataPtr[id].getPrecondIdx() > 0 && agentMetaDataPtr[id].getAge() >= 60)
                return thrust::make_pair(true, cat2_lvl);
            else
                return thrust::make_pair(false, 0.0f);
        };

        // Category elderly
        float cat3_lvl = vaccinationGroupLevel[3];
        auto cat_elderly = [agentMetaDataPtr, cat3_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
            if (agentMetaDataPtr[id].getAge() >= 60) {
                return thrust::make_pair(true, cat3_lvl); //75%: 0.89
            } else
                return thrust::make_pair(false, 0.0f);
        };

        // Category 18-59, underlying condition
        float cat4_lvl = vaccinationGroupLevel[4];
        auto cat_underlying = [agentMetaDataPtr, cat4_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
            if (agentMetaDataPtr[id].getPrecondIdx() > 0 && agentMetaDataPtr[id].getAge() >= 18
                && agentMetaDataPtr[id].getAge() < 60) {
                return thrust::make_pair(true, cat4_lvl); //75%: 0.95
            } else
                return thrust::make_pair(false, 0.0f);
        };

        // Category essential workers
        float cat5_lvl = vaccinationGroupLevel[5];
        auto cat_essential = [locationOffsetPtr, possibleTypesPtr, essentialPtr, possibleLocationsPtr, cat5_lvl] HD(
                                 unsigned id) -> thrust::pair<bool, float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
                if (possibleTypesPtr[idx] == 4
                    && essentialPtr[possibleLocationsPtr[idx]] == 1)
                    return thrust::make_pair(true, cat5_lvl); //75%: 0.85
            }
            return thrust::make_pair(false, 0.0f);
        };

        // Category school workers
        float cat6_lvl = vaccinationGroupLevel[6];
        auto cat_school = [locationOffsetPtr, possibleTypesPtr, locationTypePtr, possibleLocationsPtr, cat6_lvl] HD(
                              unsigned id) -> thrust::pair<bool, float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id + 1]; idx++) {
                if (possibleTypesPtr[idx] == 4 && locationTypePtr[possibleLocationsPtr[idx]] == 3) {
                    return thrust::make_pair(true, cat6_lvl); //75%: 0.8
                }
            }
            return thrust::make_pair(false, 0.0f);
        };

        // Category over 18-59
        float cat7_lvl = vaccinationGroupLevel[7];
        auto cat_adult = [agentMetaDataPtr, cat7_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
            if (agentMetaDataPtr[id].getAge() > 17 && agentMetaDataPtr[id].getAge() < 60) {
                return thrust::make_pair(true, cat7_lvl); //75%: 0.88
            } else
                return thrust::make_pair(false, 0.0f);
        };

        // Category over 12-18
        float cat8_lvl = vaccinationGroupLevel[8];
        auto cat_child = [agentMetaDataPtr, cat8_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
            if (agentMetaDataPtr[id].getAge() >= 12 && agentMetaDataPtr[id].getAge() < 18) {
                return thrust::make_pair(true, cat8_lvl); //75%: 0.6
            } else
                return thrust::make_pair(false, 0.0f);
        };

        // Category over 5-12
        float cat9_lvl = vaccinationGroupLevel[9];
        auto cat_child5 = [agentMetaDataPtr, cat9_lvl] HD(unsigned id) -> thrust::pair<bool, float> {
            if (agentMetaDataPtr[id].getAge() >= 5 && agentMetaDataPtr[id].getAge() < 12) {
                return thrust::make_pair(true, cat9_lvl); //75%: 0.4
            } else
                return thrust::make_pair(false, 0.0f);
        };

        uint8_t lorder[numberOfCategories];
        for (unsigned i = 0; i < numberOfCategories; i++) lorder[i] = vaccinationOrder[i];

        for (unsigned i = 0; i < numberOfCategories; i++) {
            auto it = std::find(vaccinationOrder.begin(), vaccinationOrder.end(), i + 1);
            while (it != vaccinationOrder.end()) {
                auto it = std::find(vaccinationOrder.begin(), vaccinationOrder.end(), i + 1);
                if (it == vaccinationOrder.end()) break;
                *it = -1 * (*it);
                unsigned groupIdx = std::distance(vaccinationOrder.begin(), it);
                // Figure out which round of immunizations agent belongs to, and decide if agent wants to get it.
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
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[0];
                            else
                                round = (uint8_t)-1;
                            return;
                        }

                        ret = cat_nursery(id);
                        if (groupIdx == 1 && ret.first && round == 0) {
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[1];
                            else
                                round = (uint8_t)-1;
                            return;
                        }

                        ret = cat_elderly(id);
                        if (groupIdx == 2 && ret.first && round == 0) {
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[2];
                            else
                                round = (uint8_t)-1;
                            return;
                        }

                        ret = cat_underlying(id);
                        if (groupIdx == 3 && ret.first && round == 0) {
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[3];
                            else
                                round = (uint8_t)-1;
                            return;
                        }

                        ret = cat_essential(id);
                        if (groupIdx == 4 && ret.first && round == 0) {
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[4];
                            else
                                round = (uint8_t)-1;
                            return;
                        }

                        ret = cat_adult(id);
                        if (groupIdx == 5 && ret.first && round == 0) {
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[5];
                            else
                                round = (uint8_t)-1;
                            return;
                        }

                        ret = cat_elderly_underlying(id);
                        if (groupIdx == 6 && ret.first && round == 0) {
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[6];
                            else
                                round = (uint8_t)-1;
                            return;
                        }

                        ret = cat_school(id);
                        if (groupIdx == 7 && ret.first && round == 0) {
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[7];
                            else
                                round = (uint8_t)-1;
                            return;
                        }

                        ret = cat_child(id);
                        if (groupIdx == 8 && ret.first && round == 0) {
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[8];
                            else
                                round = (uint8_t)-1;
                            return;
                        }

                        ret = cat_child5(id);
                        if (groupIdx == 9 && ret.first && round == 0) {
                            if (RandomGenerator::randomUnit() < ret.second)
                                round = lorder[9];
                            else
                                round = (uint8_t)-1;
                            return;
                        }
                    });
            }
        }
        for (unsigned i = 0; i < numberOfCategories; i++) vaccinationOrder[i] = -1 * vaccinationOrder[i];
    }
    void update(Timehandler& simTime, unsigned timeStep) {
        unsigned timestamp = simTime.getTimestamp();
        if (timestamp == 0) timestamp++;// First day already immunizing, then we sohuld not set immunizedTimestamp to 0

        // Update immunity based on days since immunization
        float immunizationEfficiencyInfectionLocal[3*MAX_STRAINS];
        float immunizationEfficiencyProgressionLocal[3*MAX_STRAINS];
        float acquired[2*MAX_STRAINS];
        for (int i = 0; i < immunizationEfficiencyInfection.size(); i++) {
            immunizationEfficiencyInfectionLocal[i] = immunizationEfficiencyInfection[i];
            immunizationEfficiencyProgressionLocal[i] = immunizationEfficiencyProgression[i];
            acquired[i] = acquiredMultiplier[i];
        }
        // float waning[] = {1.00,1.00,1.00,0.83,0.74,0.49,0.18};
        float waning[] = {1.00,1.00,1.00,0.83,0.74,0.65,0.6};

        unsigned numVariantsLocal = numVariants;
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(sim->agents->PPValues.begin(), sim->agents->agentStats.begin(), sim->agents->agentMetaData.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(sim->agents->PPValues.end(), sim->agents->agentStats.end(), sim->agents->agentMetaData.end())),
            [timeStep, timestamp, immunizationEfficiencyInfectionLocal, immunizationEfficiencyProgressionLocal, numVariantsLocal, waning, acquired] HD(thrust::tuple<typename Simulation::PPState_t&, AgentStats&, typename Simulation::AgentMeta_t&> tup) {
                if (thrust::get<0>(tup).isInfectious() > 0 &&
                    thrust::get<1>(tup).infectedCount > 1 /*&&
                    thrust::get<0>(tup).getVariant() < 2*/) {
                        thrust::get<0>(tup).reduceInfectiousness(0.65);
                }
                float susceptibilityLocal[MAX_STRAINS];
                for (int i = 0; i < numVariantsLocal; i ++) {
                    susceptibilityLocal[i] = 1.0;
                }
                //Acquired immunity with waning
                if (thrust::get<0>(tup).getStateIdx() == 0 && //susceptible
                    thrust::get<1>(tup).infectedCount > 0) {
                        unsigned months = (timestamp - thrust::get<1>(tup).infectedTimestamp) / (24 * 60 * 30 / timeStep);
                        if (months > 6) months = 6;
                        for (int i = 0; i < numVariantsLocal; i ++) {
                            if (i == thrust::get<1>(tup).variant || ((i==3 || i==4) && (thrust::get<1>(tup).variant==3 || thrust::get<1>(tup).variant==4))) {
                                float mul = (i == thrust::get<1>(tup).variant) ? 0.95 : 0.65;
                                susceptibilityLocal[i] = MIN(susceptibilityLocal[i],1.0f-(waning[months]*mul));
                                thrust::get<2>(tup).setScalingSymptoms(MIN(thrust::get<2>(tup).getScalingSymptoms(i),0.1f), i);
                            } else {
                                // if (thrust::get<1>(tup).worstState == 2) {//asymptomatic
                                //     susceptibilityLocal[i] = MIN(susceptibilityLocal[i],1.0f-(waning[months]*acquired[2*i]/2.0f));
                                //     thrust::get<2>(tup).setScalingSymptoms(MIN(thrust::get<2>(tup).getScalingSymptoms(i),MIN(1.0,acquired[2*i+1]*2)), i);
                                // } else {
                                    susceptibilityLocal[i] = MIN(susceptibilityLocal[i],1.0f-(waning[months]*acquired[2*i]));
                                    thrust::get<2>(tup).setScalingSymptoms(MIN(thrust::get<2>(tup).getScalingSymptoms(i),acquired[2*i+1]), i);
                                // }
                            }
                        }
                }

                // Otherwise get more immune after days since immunization
                unsigned daysSinceImmunization = (timestamp - thrust::get<1>(tup).immunizationTimestamp) / (24 * 60 / timeStep);
                for (int i = 0; i < numVariantsLocal; i ++) {
                    unsigned months = daysSinceImmunization < 30 ? 0 : (daysSinceImmunization/30-1);
                    if (months > 6) months = 6;

                    if (thrust::get<1>(tup).immunizationCount > 1 && daysSinceImmunization >=7) {
                        susceptibilityLocal[i] = MIN(susceptibilityLocal[i],1.0f-(waning[months]*immunizationEfficiencyInfectionLocal[3*i+2]));
                        thrust::get<2>(tup).setScalingSymptoms(MIN(thrust::get<2>(tup).getScalingSymptoms(i),immunizationEfficiencyProgressionLocal[3*i+2]), i);
                    } else if (thrust::get<1>(tup).immunizationTimestamp > 0 && daysSinceImmunization >= 28 || thrust::get<1>(tup).immunizationCount > 1) {
                        susceptibilityLocal[i] = MIN(susceptibilityLocal[i],1.0f-(waning[months]*immunizationEfficiencyInfectionLocal[3*i+1]));
                        thrust::get<2>(tup).setScalingSymptoms(MIN(thrust::get<2>(tup).getScalingSymptoms(i),immunizationEfficiencyProgressionLocal[3*i+1]), i);
                    } else if (thrust::get<1>(tup).immunizationTimestamp > 0 && daysSinceImmunization >= 12) {
                        susceptibilityLocal[i] = MIN(susceptibilityLocal[i],1.0f-immunizationEfficiencyInfectionLocal[3*i]);
                        thrust::get<2>(tup).setScalingSymptoms(MIN(thrust::get<2>(tup).getScalingSymptoms(i),immunizationEfficiencyProgressionLocal[3*i]), i);
                    }
                }

                if (thrust::get<1>(tup).immunizationCount) {
                        //If agent is infected, make them less infectious TODO: external parameter
                        if (thrust::get<0>(tup).isInfectious() /*&& thrust::get<0>(tup).getVariant() < 2*/)
                            thrust::get<0>(tup).reduceInfectiousness(0.65);
                }

                for (int i = 0; i < numVariantsLocal; i ++) {
                    if (thrust::get<0>(tup).getStateIdx() == 0) //susceptible
                    //TODO: can suscxeptibility set anywhere else?
                        thrust::get<0>(tup).setSusceptible(susceptibilityLocal[i],i);
                }

            });

        //boosters - fully random distribution
        if (numberOfBoostersToday(simTime)) {
            unsigned available = numberOfBoostersToday(simTime);
            //Count how many people eligible (4 months since last infection or previous vaccine)
            unsigned count = thrust::count_if(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        sim->agents->agentStats.begin(), sim->agents->PPValues.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        sim->agents->agentStats.end(), sim->agents->PPValues.end())),
                    [timeStep, timestamp] HD(
                        thrust::tuple<AgentStats, typename Simulation::PPState_t> tup) {
                        if (thrust::get<0>(tup).immunizationTimestamp > 0 &&
                            thrust::get<0>(tup).immunizationCount == 1 && //For now, only 1 booster
                            (timestamp - thrust::get<0>(tup).immunizationTimestamp) / (24 * 60 / timeStep) > 4 * 30
                            && thrust::get<1>(tup).getWBState() == states::WBStates::W
                            && (timestamp >= (24 * 60 / timeStep) * 3 * 30
                                    && thrust::get<0>(tup).diagnosedTimestamp < timestamp - (24 * 60 / timeStep) * 4 * 30)) {
                            return true;
                        }
                        return false;
                    });

            // Probability of each getting vaccinated today
            float prob;
            if (count < available)
                prob = 1.0;
            else
                prob = (float)available / (float)count;
            
            thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        sim->agents->agentStats.begin(), sim->agents->PPValues.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        sim->agents->agentStats.end(), sim->agents->PPValues.end())),
                    [timeStep, timestamp, prob] HD(
                        thrust::tuple<AgentStats&, typename Simulation::PPState_t&> tup) {
                        if (thrust::get<0>(tup).immunizationTimestamp > 0 &&
                            (timestamp - thrust::get<0>(tup).immunizationTimestamp) / (24 * 60 / timeStep) > 120 
                            && thrust::get<1>(tup).getWBState() == states::WBStates::W
                            && (timestamp >= (24 * 60 / timeStep) * 3 * 30
                                    && thrust::get<0>(tup).diagnosedTimestamp < timestamp - (24 * 60 / timeStep) * 4 * 30)) {
                            if (prob == 1.0f || RandomGenerator::randomUnit() < prob) {
                                thrust::get<0>(tup).immunizationTimestamp = timestamp;
                                thrust::get<0>(tup).immunizationCount++;
                            }
                        }
                    });
        }

        this->immunizedToday = 0;
        // no vaccines today, or everybody already immunized, return
        if (numberOfVaccinesToday(simTime) == 0 || currentCategory > numberOfCategories) return;

        unsigned available = numberOfVaccinesToday(simTime);
        while (available > 0 && currentCategory < numberOfCategories) {
            // Count number of eligible in current group
            unsigned count = 0;
            while (count == 0 && currentCategory <= numberOfCategories) {
                unsigned currentCategoryLocal = currentCategory + 1;// agents' categories start at 1
                unsigned numberOfCategoriesLocal = numberOfCategories + 1;
                count = thrust::count_if(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        immunizationRound.begin(), sim->agents->agentStats.begin(), sim->agents->PPValues.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        immunizationRound.end(), sim->agents->agentStats.end(), sim->agents->PPValues.end())),
                    [currentCategoryLocal, numberOfCategoriesLocal, timeStep, timestamp] HD(
                        thrust::tuple<uint8_t, AgentStats, typename Simulation::PPState_t> tup) {
                        if ((thrust::get<0>(tup) == currentCategoryLocal || currentCategoryLocal==numberOfCategoriesLocal) &&
                            thrust::get<1>(tup).immunizationTimestamp == 0
                            && thrust::get<2>(tup).getWBState() == states::WBStates::W
                            && ((timestamp < (24 * 60 / timeStep) * 3 * 30 && thrust::get<1>(tup).diagnosedTimestamp == 0)
                                || (timestamp >= (24 * 60 / timeStep) * 3 * 30
                                    && thrust::get<1>(tup).diagnosedTimestamp < timestamp - (24 * 60 / timeStep) * 3 * 30))) {
                            return true;
                        }
                        return false;
                    });
                if (count == 0) currentCategory++;
            }


            // Probability of each getting vaccinated today
            float prob;
            if (count < available)
                prob = 1.0;
            else
                prob = (float)available / (float)count;
            // printf("count %d avail %d category %d\n", count, available, currentCategory);

            // immunize available number of people in current category
            unsigned currentCategoryLocal = currentCategory + 1;
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                                 immunizationRound.begin(), sim->agents->agentStats.begin(), sim->agents->PPValues.begin())),
                thrust::make_zip_iterator(
                    thrust::make_tuple(immunizationRound.end(), sim->agents->agentStats.end(), sim->agents->PPValues.end())),
                [currentCategoryLocal, timeStep, timestamp, prob] HD(
                    thrust::tuple<uint8_t&, AgentStats&, typename Simulation::PPState_t&> tup) {
                    if (thrust::get<0>(tup) == currentCategoryLocal &&// TODO how many days since diagnosis?
                        thrust::get<1>(tup).immunizationTimestamp == 0
                        && thrust::get<2>(tup).getWBState() == states::WBStates::W
                        && ((timestamp < (24 * 60 / timeStep) * 3 * 30 && thrust::get<1>(tup).diagnosedTimestamp == 0)
                            || (timestamp >= (24 * 60 / timeStep) * 3 * 30
                                && thrust::get<1>(tup).diagnosedTimestamp < timestamp - (24 * 60 / timeStep) * 3 * 30))) {
                        if (prob == 1.0f || RandomGenerator::randomUnit() < prob)
                            thrust::get<1>(tup).immunizationTimestamp = timestamp;
                            thrust::get<1>(tup).immunizationCount = 1;
                    }
                });
            if (diagnosticLevel > 0) {
                auto it = std::find(vaccinationOrder.begin(), vaccinationOrder.end(), currentCategory + 1);
                unsigned groupIdx = std::distance(vaccinationOrder.begin(), it);
                std::cout << "Immunized " << (count < available ? count : available) << " people from group " << groupIdx
                          << std::endl;
            }
            this->immunizedToday += (count < available ? count : available);
            // subtract from available
            if (count < available)
                available -= count;
            else
                available = 0;
        }
    }
};