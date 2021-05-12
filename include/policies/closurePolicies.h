#pragma once
#include <iostream>
#include "timeHandler.h"
#include "datatypes.h"
#include "cxxopts.hpp"
#include "operators.h"
#include "locationTypesFormat.h"
namespace ClosureHelpers {
    std::vector<std::string> splitHeader(std::string &header) {
        std::stringstream ss(header);
        std::string arg;
        std::vector<std::string> params;
        for (char i; ss >> i;) {
            arg.push_back(i);    
            if (ss.peek() == '\t') {
                if (arg.length()>0) {
                    params.push_back(arg);
                    arg.clear();
                }
                ss.ignore();
            }
        }
        if (arg.length()>0) {
            params.push_back(arg);
            arg.clear();
        }
        return params;
    }
    int mod(int a, int b)
    {
        int r = a % b;
        return r < 0 ? r + b : r;
    }
}

template<typename SimulationType>
class NoClosure {
public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data, const parser::ClosureRules& rules, std::string header) {}

    void midnight(Timehandler simTime, unsigned timeStep, std::vector<unsigned> &stats) {}
    void step(Timehandler simTime, unsigned timeStep) {}
};

template<typename SimulationType>
class RuleClosure {
    public:
    std::vector<std::string> header;
    class GlobalCondition {
        public:
        std::vector<unsigned> headerPos;
        std::vector<double> history;
        int pos;
        bool active;
        std::function<bool(GlobalCondition *, std::vector<unsigned>&)> condition;
        GlobalCondition(std::vector<unsigned> h, bool a, std::function<bool(GlobalCondition *, std::vector<unsigned>&)> r) : headerPos(h), active(a), condition(r), pos(0) {}
    };
    std::vector<GlobalCondition> globalConditions;
    class Rule {
        public:
        std::string name;
        std::vector<GlobalCondition *> conditions;
        std::function<void(Rule *)> rule;
        bool previousOpenState;
        Rule(std::string n, std::vector<GlobalCondition *> c, std::function<void(Rule *)> r) : name (n), conditions(c), rule(r), previousOpenState(true) {}
    };
    std::vector<Rule> rules;
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("enableClosures",
            "Enable(1)/disable(0) closure rules defined in closureRules.json",
            cxxopts::value<unsigned>()->default_value("1"));
    }
    unsigned enableClosures;
    unsigned diagnosticLevel=0;
    void initializeArgs(const cxxopts::ParseResult& result) {
        enableClosures = result["enableClosures"].as<unsigned>();
        diagnosticLevel = result["diags"].as<unsigned>();
    }
    void init(const parser::LocationTypes& data, const parser::ClosureRules& rules, std::string header) {
        this->header = ClosureHelpers::splitHeader(header);

        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();
        unsigned diags = diagnosticLevel;

        std::vector<unsigned> hospitalStates;
        for (unsigned i = 0; i < this->header.size(); i++)
            if (this->header[i].find("_h")!=std::string::npos) hospitalStates.push_back(i);

        globalConditions.reserve(rules.rules.size());
        this->rules.reserve(rules.rules.size());

        for (const parser::ClosureRules::Rule& rule : rules.rules) {

            //check if masks/closures are enabled
            if (!enableClosures && !rule.name.compare("Masks")==0 
                && !rule.name.compare("Curfew")==0 
                && !rule.name.compare("HolidayMode")==0 
                && !rule.name.compare("SchoolAgeRestriction")==0
                && !rule.name.compare("ExposeToMutation")==0
                && !rule.name.compare("ReduceMovement")==0) continue;

            //Create condition

            int closeAfter = rule.closeAfter; 
            int openAfter = rule.openAfter;
            double threshold = rule.threshold;
            if (rule.conditionType.compare("afterDays")==0) {
                std::vector<unsigned> none;
                if (closeAfter >=0 || openAfter<=0) throw CustomErrors("For closure rule 'afterDays', closeAfter must be -1, openAfter must be >0");
                globalConditions.emplace_back(none, false, [openAfter,threshold](GlobalCondition* c, std::vector<unsigned>& stats){
                     int day = c->history[0]++; return (day>=threshold && day < threshold + openAfter); });
                globalConditions.back().history.resize(1,0.0);
            } else if (rule.conditionType.compare("hospitalizedFraction")==0) {
                if (closeAfter <=0 || openAfter<=0) throw CustomErrors("For closure rule 'hospitalizedFraction', closeAfter and openAfter must be >0");
                globalConditions.emplace_back(hospitalStates, 0, [threshold, openAfter, closeAfter,numberOfAgents](GlobalCondition* c, std::vector<unsigned>& stats){
                    //calculate fraction
                    double accum = 0.0;
                    for (unsigned i = 0; i < c->headerPos.size(); i++) accum += stats[c->headerPos[i]];
                    double value = accum/(double)numberOfAgents;
                    //insert into history
                    c->history[c->pos] = value;
                    c->pos = (c->pos+1)%(c->history.size());
                    //check if above threshold, and if so has it been for the last closeAfter days
                    if (value > threshold) {
                        bool wasBelow = false;
                        for (unsigned i = ClosureHelpers::mod(c->pos-closeAfter,c->history.size()); i != c->pos; i=(i+1)%c->history.size()) {
                            wasBelow |= c->history[i] < threshold;
                        }
                        if (!wasBelow) return true;
                        else return c->active;
                    } else {
                        //below threshold for openAfter days
                        bool wasAbove = false;
                        for (unsigned i = ClosureHelpers::mod(c->pos-openAfter,c->history.size()); i != c->pos; i=(i+1)%c->history.size()) {
                            wasAbove |= c->history[i] >= threshold;
                        }
                        if (!wasAbove) return false;
                        else return c->active;
                    }
                });
                globalConditions.back().history.resize(std::max(closeAfter, openAfter)+2,0.0);
            } else if (rule.conditionType.compare("newDeadFraction")==0) {
                if (closeAfter <=0 || openAfter<=0) throw CustomErrors("For closure rule 'newDeadFraction', closeAfter and openAfter must be >0");
                std::vector<unsigned> deadState;
                for (unsigned i = 0; i < this->header.size(); i++)
                    if (this->header[i].compare("D1")==0) deadState.push_back(i);
                deadState.push_back(0);
                globalConditions.emplace_back(deadState, 0, [threshold, openAfter, closeAfter,numberOfAgents](GlobalCondition* c, std::vector<unsigned>& stats){
                    double value =  (stats[c->headerPos[0]]-c->headerPos[1])/(double)numberOfAgents; //new dead is number of dead - last time
                    c->headerPos[1] = stats[c->headerPos[0]]; //save previous value
                    //insert into history
                    c->history[c->pos] = value;
                    c->pos = (c->pos+1)%(c->history.size());
                    //check if above threshold, and if so has it been for the last closeAfter days
                    if (value > threshold) {
                        bool wasBelow = false;
                        for (unsigned i = ClosureHelpers::mod(c->pos-closeAfter,c->history.size()); i != c->pos; i=(i+1)%c->history.size()) {
                            wasBelow |= c->history[i] < threshold;
                        }
                        if (!wasBelow) return true;
                        else return c->active;
                    } else {
                        //below threshold for openAfter days
                        bool wasAbove = false;
                        for (unsigned i = ClosureHelpers::mod(c->pos-openAfter,c->history.size()); i != c->pos; i=(i+1)%c->history.size()) {
                            wasAbove |= c->history[i] >= threshold;
                        }
                        if (!wasAbove) return false;
                        else return c->active;
                    }
                });
                globalConditions.back().history.resize(std::max(closeAfter, openAfter)+2,0.0);
            } else {
                throw CustomErrors("Unknown closure type "+rule.conditionType);
            }

            if (rule.name.compare("Masks")==0) {
                //Masks
                thrust::device_vector<double>& locInfectiousness = realThis->locs->infectiousness;
                thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locTypes = realThis->locs->locType;
                unsigned homeType = data.home;
                double maskCoefficient2 = std::stod(rule.parameter);
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                this->rules.emplace_back(rule.name, conds, [&,homeType,maskCoefficient2,diags](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active;}
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(locTypes.begin(), locInfectiousness.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(locTypes.end(), locInfectiousness.end())),
                                        [maskCoefficient2,homeType,shouldBeOpen]HD(thrust::tuple<typename SimulationType::TypeOfLocation_t&, double&> tup)
                                        {
                                            auto& type = thrust::get<0>(tup);
                                            auto& infectiousness = thrust::get<1>(tup);
                                            if (type != homeType) {
                                                if (shouldBeOpen) infectiousness = infectiousness / maskCoefficient2;
                                                else infectiousness = infectiousness * maskCoefficient2;
                                            }
                                        });
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("Masks %s with %g multiplier\n", (int)shouldBeOpen ? "off": "on", maskCoefficient2);
                    }
                });
            } else if (rule.name.compare("Curfew")==0) {
                unsigned curfewBegin;
                unsigned curfewEnd;
                if (rule.parameter.length()==9) {
                    unsigned bhours = atoi(rule.parameter.substr(0,2).c_str());
                    unsigned bminutes = atoi(rule.parameter.substr(2,2).c_str());
                    curfewBegin = bhours*60+bminutes;
                    unsigned ehours = atoi(rule.parameter.substr(5,2).c_str());
                    unsigned eminutes = atoi(rule.parameter.substr(7,2).c_str());
                    curfewEnd = ehours*60+eminutes;
                } else if (rule.parameter.length()>0) throw CustomErrors("curfew parameter string needs to be exactly 9 characters long");
                
                //Curfew
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                auto realThis = static_cast<SimulationType*>(this);
                this->rules.emplace_back(rule.name, conds, [&, realThis,diags,curfewBegin,curfewEnd](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active;}
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        realThis->toggleCurfew(close,curfewBegin,curfewEnd);
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("Curfew %s\n", (int)shouldBeOpen ? "disabled": "enabled");
                    }
                });
            } else if (rule.name.compare("HolidayMode")==0) {
                //Curfew
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                auto realThis = static_cast<SimulationType*>(this);
                this->rules.emplace_back(rule.name, conds, [&, realThis,diags](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active;}
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        realThis->toggleHolidayMode(close);
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("Holiday mode %s\n", (int)shouldBeOpen ? "disabled": "enabled");
                    }
                });
            } else if (rule.name.compare("SchoolAgeRestriction")==0) {
                //Restrict school going age
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                auto realThis = static_cast<SimulationType*>(this);
                unsigned ageLimit = std::stoi(rule.parameter);
                this->rules.emplace_back(rule.name, conds, [&, realThis,diags,ageLimit](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active;}
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        realThis->setSchoolAgeRestriction(close?ageLimit:99);
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("School age restriction %s - only under %d years go to school\n", (int)shouldBeOpen ? "disabled": "enabled", ageLimit);
                    }
                });
            } else if (rule.name.compare("ExposeToMutation")==0) {
                //expose population to mutated version
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                auto realThis = static_cast<SimulationType*>(this);
                double fraction = std::stod(rule.parameter);
                this->rules.emplace_back(rule.name, conds, [&, realThis,diags,fraction](Rule *r) {
                    bool expose = true;
                    for (GlobalCondition *c : r->conditions) {expose = expose && c->active;}
                    if (expose) {
                        thrust::device_vector<unsigned> exposed(realThis->agents->PPValues.size(),0);
                        unsigned timestamp = realThis->getSimTime().getTimestamp();
                        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(realThis->agents->PPValues.begin(),
                                                                                    realThis->agents->location.begin(),
                                                                                    exposed.begin(),
                                                                                    realThis->agents->agentStats.begin())) ,
                                        thrust::make_zip_iterator(thrust::make_tuple(realThis->agents->PPValues.end(),
                                                                                    realThis->agents->location.end(),
                                                                                    exposed.end(),
                                                                                    realThis->agents->agentStats.end())) ,
                                [fraction,timestamp]HD(thrust::tuple<typename SimulationType::PPState_t &, unsigned&, unsigned&, AgentStats&> tup) {
                                    auto &state = thrust::get<0>(tup);
                                    auto &agentLocation = thrust::get<1>(tup);
                                    auto &exposed = thrust::get<2>(tup);
                                    auto &agentStat = thrust::get<3>(tup);
                                    if (state.getSusceptible()>0.0f && RandomGenerator::randomUnit() < fraction*state.getSusceptible()) {
                                        state.gotInfected(1);
                                        agentStat.infectedTimestamp = timestamp;
                                        agentStat.infectedLocation = agentLocation;
                                        agentStat.worstState = state.getStateIdx();
                                        agentStat.worstStateTimestamp = timestamp;
                                        agentStat.variant = 1;
                                        exposed = 1;
                                    }
                                });
                        unsigned count = thrust::count(exposed.begin(), exposed.end(), unsigned(1));
                        if (diags>0) printf("Exposed %d people to mutation\n", count);
                    }
                });
            } else if (rule.name.compare("ReduceMovement")==0) {
                //Masks
                unsigned homeType = data.home;
                AgentTypeList &agentTypes = realThis->agents->agentTypes;
                double fraction = std::stod(rule.parameter);
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                this->rules.emplace_back(rule.name, conds, [homeType,fraction,diags,&agentTypes](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active;}
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        if (shouldBeOpen)
                            agentTypes.unsetStayHome(fraction, homeType);
                        else
                            agentTypes.setStayHome(fraction, homeType);
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("Reduced movement %s with %g multiplier\n", (int)shouldBeOpen ? "off": "on", fraction);
                    }
                });
            } else { //Not masks, curfew, holiday
                //Create rule
                thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locTypes = realThis->locs->locType;
                thrust::device_vector<bool>& locationOpenState = realThis->locs->states;
                thrust::device_vector<uint8_t>& locationEssential = realThis->locs->essential;
                const std::vector<int> &locationTypesToClose = rule.locationTypesToClose;

                //Create small fixed size array for listing types to close that can be captured properly 
                //typename SimulationType::TypeOfLocation_t fixListArr[10];
                std::array<unsigned, 10> fixListArr;
                if (locationTypesToClose.size()>=10) throw CustomErrors("Error, Closure Rule " + rule.name+ " has over 10 location types to close, please increase implementation limit");
                for (unsigned i = 0; i < locationTypesToClose.size(); i++) {
                    fixListArr[i] = locationTypesToClose[i];
                }
                for (unsigned i = locationTypesToClose.size(); i < 10; i++) fixListArr[i] = (typename SimulationType::TypeOfLocation_t)-1;

                //printf("cond %p\n",&globalConditions[globalConditions.size()-1]);
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                this->rules.emplace_back(rule.name, conds, [&,fixListArr,diags](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active; /*printf("rule %s cond %p\n", r->name.c_str(), c);*/}
                    //printf("Rule %s %d\n", r->name.c_str(), close ? 1 : 0);
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(locTypes.begin(), locationOpenState.begin(), locationEssential.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(locTypes.end(), locationOpenState.end(), locationEssential.end())),
                                        [fixListArr,shouldBeOpen]HD(thrust::tuple<typename SimulationType::TypeOfLocation_t&, bool&, uint8_t&> tup)
                                        {
                                            auto& type = thrust::get<0>(tup);
                                            auto& isOpen = thrust::get<1>(tup);
                                            auto& isEssential = thrust::get<2>(tup);
                                            if (isEssential==1) return;
                                            for (unsigned i = 0; i < 10; i++)
                                                if (type == fixListArr[i])
                                                    isOpen = shouldBeOpen;
                                                else if ((typename SimulationType::TypeOfLocation_t)-1 == fixListArr[i]) break;
                                        });
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("Rule %s %s\n", r->name.c_str(), (int)shouldBeOpen ? "disabled": "enabled");
                    }
                });
            } 
        }
    }

    void midnight(Timehandler simTime, unsigned timeStep, std::vector<unsigned> &stats) {
        for (GlobalCondition &c : globalConditions) {
            c.active = c.condition(&c,stats);
        }
        for (Rule &r : rules) {
            r.rule(&r);
        }
        
    }
    void step(Timehandler simTime, unsigned timeStep) {}
};
