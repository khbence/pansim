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


template <class Simulation>
class Immunization {
    Simulation *sim;
    thrust::device_vector<uint8_t> immunizationRound;
    unsigned currentCategory = 0;
#define numberOfCategories 9
    std::vector<uint8_t> vaccinationOrder;
    unsigned startAfterDay = 0;
    unsigned dailyDoses = 0;
    unsigned diagnosticLevel = 0;

    unsigned numberOfVaccinesToday(Timehandler& simTime) {
        //float availPerWeek[] = {1115.178518, 486.88636, 895.271552, 1876.9132, 1955.46154, 4943.59527, 8544.33033, 8563.97919, 9160.88972, 9612.38930, 10252.4613, 10276.0211, 9361.10071, 9553.48985, 9977.56585, 9722.31922, 8823.13674, 8823.13674, 8756.36833, 8783.88615, 8799.57697, 5717.15701, 5705.37712, 5670.0374, 5662.16849, 5799.61623, 5795.70531, 5772.14553, 3852.02365, 3043.12224, 3031.34235, 934.522142};
        //float availPerWeek[] = {1115.178518, 486.88636, 895.271552, 1876.9132, 1955.46154, 2471.797635, 4272.165165, 4281.989595, 4580.44486, 4806.19465, 5126.23065, 5138.01055, 4680.550355, 4776.744925, 4988.782925, 4861.15961, 4411.56837, 4411.56837, 4378.184165, 4391.943075, 4399.788485, 4399.788485, 4399.788485, 4399.788485, 4399.788485, 4399.788485, 4399.788485, 4399.788485, 4399.788485, 4399.788485, 4399.788485, 4399.788485};
        // float availPerWeek[] = {1260.0, 1260.0,1260.0, 1260.0, 1260.0, 1260.0, 1260.0, 1260.0, //8 weeks of 0.1%
        //                         921*7.0*1.0,  1126*7.0*1.0, 768*7.0*1.0, 1060*7.0*1.0, 821*7.0*1.0,  1506*7.0*1.0, 1506*7.0*1.0, 1506*7.0*1.0, 1506*7.0*1.0, 1973*7.0, 1973*7.0, 1973*7.0, 1973*7.0}; //latest prediction
                                //3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780,3780}; //then 
        if (simTime.getTimestamp()/(24*60/simTime.getTimeStep()) >= startAfterDay) {
            // unsigned day = simTime.getTimestamp()/(24*60/simTime.getTimeStep())-startAfterDay;
            // unsigned week = day/7;
            // return availPerWeek[week>21?21:week]/7.0;
            return dailyDoses;
        }
        else return 0;
    }

    public:
    unsigned immunizedToday = 0;

    Immunization (Simulation *s) {
        this->sim = s;
    }
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("immunizationStart",
            "number of days into simulation when immunization starts",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(0))))
            ("immunizationsPerDay",
            "number of immunizations per day",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(0))))
            ("immunizationOrder",
            "Order of immunization (starting at 1, 0 to skip) for agents in different categories health workers, nursery home worker/resident, 60+, 18-60 with underlying condition, essential worker, 18+, 60+underlying, school teachers, children",
            cxxopts::value<std::string>()->default_value("1,2,3,4,5,6,0,0,0"));
    }

    void initializeArgs(const cxxopts::ParseResult& result) {
        startAfterDay = result["immunizationStart"].as<unsigned>();
        dailyDoses = result["immunizationsPerDay"].as<unsigned>();
        try {
            diagnosticLevel = result["diags"].as<unsigned>();
        } catch (std::exception &e) {}

        std::string orderString = result["immunizationOrder"].as<std::string>();
        std::stringstream ss(orderString);
        std::string arg;
        for (char i; ss >> i;) {
            arg.push_back(i);    
            if (ss.peek() == ',') {
                if (arg.length()>0 && isdigit(arg[0])) {
                    vaccinationOrder.push_back(atoi(arg.c_str()));
                    arg.clear();
                }
                ss.ignore();
            }
        }
        if (arg.length()>0 && isdigit(arg[0])) {
            vaccinationOrder.push_back(atoi(arg.c_str()));
            arg.clear();
        }
        if (vaccinationOrder.size()!=numberOfCategories) throw CustomErrors("immunizationOrder mush have exactly "+std::to_string(numberOfCategories)+" values");
        for (int i = 0; i < numberOfCategories; i++)
            if (vaccinationOrder[i]>numberOfCategories) throw CustomErrors("immunizationOrder values have to be  less or equal to "+std::to_string(numberOfCategories));

    }

    void initCategories() {
        immunizationRound.resize(sim->agents->PPValues.size(),0);
        
        auto *agentMetaDataPtr = thrust::raw_pointer_cast(sim->agents->agentMetaData.data());
        auto *locationOffsetPtr = thrust::raw_pointer_cast(sim->agents->locationOffset.data());
        auto *possibleTypesPtr = thrust::raw_pointer_cast(sim->agents->possibleTypes.data());
        auto *locationTypePtr = thrust::raw_pointer_cast(sim->locs->locType.data());
        auto *possibleLocationsPtr = thrust::raw_pointer_cast(sim->agents->possibleLocations.data());
        auto *essentialPtr = thrust::raw_pointer_cast(sim->locs->essential.data());
        

        //Figure out which category agents belong to, and determine if agent is willing to be vaccinated

        //Category health worker
        auto cat_healthworker = [locationOffsetPtr, possibleTypesPtr,possibleLocationsPtr,locationTypePtr] HD (unsigned id) -> thrust::pair<bool,float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id+1]; idx++) {
                //TODO pull these params from config
                if (possibleTypesPtr[idx] == 4 && (locationTypePtr[possibleLocationsPtr[idx]]==12 || locationTypePtr[possibleLocationsPtr[idx]]==14))
                    return thrust::make_pair(true, 0.7f);
            }
            return thrust::make_pair(false,0.0f);
        };

        //Category nursery home workers & residents
        auto cat_nursery = [locationOffsetPtr, possibleTypesPtr,locationTypePtr,possibleLocationsPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id+1]; idx++) {
                if ((possibleTypesPtr[idx] == 4 || possibleTypesPtr[idx] == 2) && locationTypePtr[possibleLocationsPtr[idx]]==22) //TODO pull these params from config
                    return thrust::make_pair(true, 0.9f);
            }
            return thrust::make_pair(false,0.0f);
        };

        //Category elderly with underlying condition
        auto cat_elderly_underlying = [agentMetaDataPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            if (agentMetaDataPtr[id].getPrecondIdx()>0 && agentMetaDataPtr[id].getAge()>=60) return thrust::make_pair(true, 0.8f);
            else return thrust::make_pair(false,0.0f);
        };

        //Category elderly
        auto cat_elderly = [agentMetaDataPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            if (agentMetaDataPtr[id].getAge()>=60) return thrust::make_pair(true, 0.7f);
            else return thrust::make_pair(false,0.0f);
        };

        //Category 18-59, underlying condition
        auto cat_underlying = [agentMetaDataPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            if (agentMetaDataPtr[id].getPrecondIdx()>0 && agentMetaDataPtr[id].getAge()>=18 && agentMetaDataPtr[id].getAge()<60) return thrust::make_pair(true, 0.8f);
            else return thrust::make_pair(false,0.0f);
        };

        //Category essential workers
        auto cat_essential = [locationOffsetPtr, possibleTypesPtr,essentialPtr,possibleLocationsPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id+1]; idx++) {
                if (possibleTypesPtr[idx] == 4 && essentialPtr[possibleLocationsPtr[idx]]==1) //TODO pull these params from config
                    return thrust::make_pair(true, 0.6f);
            }
            return thrust::make_pair(false,0.0f);
        };

        //Category school workers
        auto cat_school = [locationOffsetPtr, possibleTypesPtr,locationTypePtr,possibleLocationsPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id+1]; idx++) {
                if (possibleTypesPtr[idx] == 4 && locationTypePtr[possibleLocationsPtr[idx]]==3)
                    return thrust::make_pair(true, 0.7f);
            }
            return thrust::make_pair(false,0.0f);
        };

        //Category over 18
        auto cat_adult = [agentMetaDataPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            if (agentMetaDataPtr[id].getAge()>17) return thrust::make_pair(true, 0.6f);
            else return thrust::make_pair(false,0.0f);
        };

        //Category over 3-18
        auto cat_child = [agentMetaDataPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            if (agentMetaDataPtr[id].getAge()>=3 && agentMetaDataPtr[id].getAge()<18) return thrust::make_pair(true, 0.7f);
            else return thrust::make_pair(false,0.0f);
        };

        uint8_t lorder[numberOfCategories];
        for (unsigned i = 0; i < numberOfCategories; i++) lorder[i] = vaccinationOrder[i];

        for (unsigned currentCategory = 0; currentCategory < numberOfCategories; currentCategory++) {
            auto it = std::find(vaccinationOrder.begin(), vaccinationOrder.end(), currentCategory+1);
            if (it == vaccinationOrder.end()) continue;
            unsigned groupIdx = std::distance(vaccinationOrder.begin(),it);
            //Figure out which round of immunizations agent belongs to, and decide if agent wants to get it.
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.begin(), thrust::make_counting_iterator(0))),
                             thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.end()  , thrust::make_counting_iterator((int)immunizationRound.size()))),
                            [cat_healthworker,cat_nursery,cat_elderly,cat_underlying,cat_essential,cat_adult,cat_elderly_underlying,cat_school,cat_child,lorder,groupIdx] HD (thrust::tuple<uint8_t&, int> tup) {
                                uint8_t& round = thrust::get<0>(tup);
                                unsigned id = thrust::get<1>(tup);

                                auto ret = cat_healthworker(id);
                                if (groupIdx == 0 && ret.first && round==0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = lorder[0];
                                    else round = (uint8_t)-1;
                                    return;
                                }

                                ret = cat_nursery(id);
                                if (groupIdx == 1 && ret.first && round==0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = lorder[1];
                                    else round = (uint8_t)-1;
                                    return;
                                }

                                ret = cat_elderly(id);
                                if (groupIdx == 2 && ret.first && round==0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = lorder[2];
                                    else round = (uint8_t)-1;
                                    return;
                                }

                                ret = cat_underlying(id);
                                if (groupIdx == 3 && ret.first && round==0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = lorder[3];
                                    else round = (uint8_t)-1;
                                    return;
                                }

                                ret = cat_essential(id);
                                if (groupIdx == 4 && ret.first && round==0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = lorder[4];
                                    else round = (uint8_t)-1;
                                    return;
                                }

                                ret = cat_adult(id);
                                if (groupIdx == 5 && ret.first && round==0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = lorder[5];
                                    else round = (uint8_t)-1;
                                    return;
                                }

                                ret = cat_elderly_underlying(id);
                                if (groupIdx == 6 && ret.first && round==0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = lorder[6];
                                    else round = (uint8_t)-1;
                                    return;
                                }

                                ret = cat_school(id);
                                if (groupIdx == 7 && ret.first && round==0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = lorder[7];
                                    else round = (uint8_t)-1;
                                    return;
                                }

                                ret = cat_child(id);
                                if (groupIdx == 8 && ret.first && round==0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = lorder[8];
                                    else round = (uint8_t)-1;
                                    return;
                                }
                            }
                        );
        }
    }
    void update(Timehandler& simTime, unsigned timeStep) {
        unsigned timestamp = simTime.getTimestamp();
        if (timestamp == 0) timestamp++; //First day already immunizing, then we sohuld not set immunizedTimestamp to 0

        //Update immunity based on days since immunization
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(sim->agents->PPValues.begin(), sim->agents->agentStats.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(sim->agents->PPValues.end(), sim->agents->agentStats.end())),
                             [timeStep,timestamp]HD(thrust::tuple<typename Simulation::PPState_t&, AgentStats&> tup) {
                                 //If not immunized, or already recovered, return
                                 if (thrust::get<1>(tup).immunizationTimestamp == 0 || thrust::get<0>(tup).getSusceptible()==0.0) return;
                                 //Otherwise get more immune after days since immunization
                                 unsigned daysSinceImmunization = (timestamp-thrust::get<1>(tup).immunizationTimestamp)/(24*60/timeStep);
                                 if (daysSinceImmunization>=28) thrust::get<0>(tup).setSusceptible(0.04); //96%
                                 else if (daysSinceImmunization>=12) thrust::get<0>(tup).setSusceptible(0.48); //52%
                             });

        this->immunizedToday = 0;
        //no vaccines today, or everybody already immunized, return
        if (numberOfVaccinesToday(simTime) == 0 || currentCategory >= numberOfCategories) return;

        unsigned available = numberOfVaccinesToday(simTime);
        while (available > 0 && currentCategory < numberOfCategories) {
            //Count number of eligible in current group
            unsigned count = 0;
            while (count == 0 && currentCategory < numberOfCategories) {
                unsigned currentCategoryLocal = currentCategory+1; //agents' categories start at 1
                count = thrust::count_if(thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.begin(), sim->agents->agentStats.begin(),sim->agents->PPValues.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.end(), sim->agents->agentStats.end(),sim->agents->PPValues.end())),
                             [currentCategoryLocal,timeStep,timestamp]HD(thrust::tuple<uint8_t, AgentStats, typename Simulation::PPState_t> tup) {
                                 if (thrust::get<0>(tup) == currentCategoryLocal &&  //TODO how many days since diagnosis?
                                     thrust::get<1>(tup).immunizationTimestamp == 0 &&
                                     thrust::get<2>(tup).getWBState() == states::WBStates::W &&
                                     ((timestamp < (24*60/timeStep)*3*30 && thrust::get<1>(tup).diagnosedTimestamp == 0) ||
                                      (timestamp >= (24*60/timeStep)*3*30 && thrust::get<1>(tup).diagnosedTimestamp < timestamp - (24*60/timeStep)*3*30))) {
                                          return true;
                                      }
                                 return false;
                             });
                if (count == 0) currentCategory++;
            }
            

            //Probability of each getting vaccinated today
            float prob;
            if (count < available) prob = 1.0;
            else prob = (float)available/(float)count;
            //printf("count %d avail %d category %d\n", count, available, currentCategory);

            //immunize available number of people in current category
            unsigned currentCategoryLocal = currentCategory+1;
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.begin(), sim->agents->agentStats.begin(),sim->agents->PPValues.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.end(), sim->agents->agentStats.end(),sim->agents->PPValues.end())),
                             [currentCategoryLocal,timeStep,timestamp,prob]HD(thrust::tuple<uint8_t&, AgentStats&, typename Simulation::PPState_t&> tup) {
                                 if (thrust::get<0>(tup) == currentCategoryLocal &&  //TODO how many days since diagnosis?
                                     thrust::get<1>(tup).immunizationTimestamp == 0 &&
                                     thrust::get<2>(tup).getWBState() == states::WBStates::W &&
                                     ((timestamp < (24*60/timeStep)*3*30 && thrust::get<1>(tup).diagnosedTimestamp == 0) ||
                                      (timestamp >= (24*60/timeStep)*3*30 && thrust::get<1>(tup).diagnosedTimestamp < timestamp - (24*60/timeStep)*3*30))) {
                                          if (prob == 1.0f || RandomGenerator::randomUnit() < prob)
                                          thrust::get<1>(tup).immunizationTimestamp = timestamp;
                                      }
                             });
            if (diagnosticLevel>0) {
                auto it = std::find(vaccinationOrder.begin(), vaccinationOrder.end(), currentCategory+1);
                unsigned groupIdx = std::distance(vaccinationOrder.begin(),it);
                std::cout << "Immunized " << (count < available? count : available) << " people from group " << groupIdx << std::endl;
            }
            this->immunizedToday += (count < available? count : available);
            //subtract from available
            if (count < available) available -= count;
            else available = 0;
        }
    }
};