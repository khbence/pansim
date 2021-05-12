#include "agentMeta.h"
#include "customExceptions.h"
#include "JSONDecoder.h"
#include <limits>
#include <algorithm>

BasicAgentMeta::AgeInterval::AgeInterval(parser::Parameters::Age in)
    : symptoms(static_cast<float>(in.symptoms)), transmission(static_cast<float>(in.transmission)) {
    if (in.from < 0) { throw IOParameters::NegativeFrom(); }
    from = in.from;
    if (in.to < 0) {
        to = std::numeric_limits<decltype(to)>::max();
    } else {
        to = in.to;
    }
    if (to < from) { throw IOParameters::NegativeInterval(from, to); }
}

float BasicAgentMeta::AgeInterval::getSymptoms() const { return symptoms; }

float BasicAgentMeta::AgeInterval::getTransmission() const { return transmission; }

std::array<std::pair<char, float>, 2> BasicAgentMeta::sexScaling;
std::vector<BasicAgentMeta::AgeInterval> BasicAgentMeta::ageScaling;
std::map<std::string, float> BasicAgentMeta::preConditionScaling;

// init the three static variable with the data that coming from parameters json
// file
void BasicAgentMeta::initData(const parser::Parameters& inputData) {
    // init the scaling based in sex
    if (inputData.sex.size() != 2) { throw IOParameters::NotBinary(); }
    for (unsigned i = 0; i < sexScaling.size(); ++i) {
        if (!(inputData.sex[0].name == "F" || inputData.sex[0].name == "M")) {
            throw IOParameters::WrongGenderName();
        }
        sexScaling[i] = std::make_pair(inputData.sex[i].name[0], inputData.sex[i].symptoms);
    }
    if (sexScaling[0].first == sexScaling[1].first) { throw IOParameters::WrongGenderName(); }
    if (sexScaling[0].first == 'M') { std::swap(sexScaling[0], sexScaling[1]); }

    // init the scalings based on age
    ageScaling.reserve(inputData.age.size());
    for (auto ageInter : inputData.age) { ageScaling.emplace_back(ageInter); }
    // TODO check intervals

    // init the scaling that coming from pre-conditions
    for (const auto& cond : inputData.preCondition) {
        preConditionScaling.emplace(std::make_pair(cond.ID, cond.symptoms));
    }
}

BasicAgentMeta::BasicAgentMeta() {}

BasicAgentMeta::BasicAgentMeta(char gender, unsigned age, std::string preCondition) {
    // modify based on gender
    if (gender == 'F') {
        scalingSymptoms *= sexScaling[0].second;
        this->sex = 0;
    } else if (gender == 'M') {
        scalingSymptoms *= sexScaling[1].second;
        this->sex = 1;
    } else {
        throw IOAgents::InvalidGender(std::to_string(gender));
    }

    this->age = (uint8_t)age;
    // modify based on age
    auto it = std::find(ageScaling.begin(), ageScaling.end(), age);
    if (it == ageScaling.end()) { throw IOAgents::NotDefinedAge(age); }
    scalingSymptoms *= it->getSymptoms();
    scalingTransmission *= it->getTransmission();

    // modify based on pre-condition
    auto itMap = preConditionScaling.find(preCondition);
    if (itMap == preConditionScaling.end()) { throw IOAgents::NotDefinedCondition(preCondition); }
    preCondIdx = std::stoi(preCondition);
    scalingSymptoms *= itMap->second;
}

float HD BasicAgentMeta::getScalingSymptoms() const { return scalingSymptoms; }

float HD BasicAgentMeta::getScalingTransmission() const { return scalingTransmission; }

uint8_t HD BasicAgentMeta::getAge() const { return age; }

bool HD BasicAgentMeta::getSex() const { return sex; }

uint8_t HD BasicAgentMeta::getPrecondIdx() const { return preCondIdx; }