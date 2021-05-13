#pragma once
#include <exception>
#include <string>

class CustomErrors : public std::exception {
    std::string error;

public:
    explicit CustomErrors(std::string&& error_p) : error(std::move(error_p)) {}
    [[nodiscard]] const char* what() const noexcept override { return error.c_str(); }
};

namespace IOProgression {
    class TransitionInputError : public CustomErrors {
    protected:
        explicit TransitionInputError(std::string&& error_p)
            : CustomErrors("Transition input file error: " + error_p) {}
    };

    class WrongNumberOfStates : public TransitionInputError {
    public:
        WrongNumberOfStates(unsigned expected, unsigned got)
            : TransitionInputError("Expected " + std::to_string(expected) + " states, got "
                                   + std::to_string(got) + ".\n") {}
    };

    class WrongStateName : public TransitionInputError {
    public:
        explicit WrongStateName(const std::string& stateName)
            : TransitionInputError(stateName + " doesn't exists.\n") {}
    };

    class TooMuchBad : public TransitionInputError {
    public:
        explicit TooMuchBad(unsigned state)
            : TransitionInputError(
                std::to_string(state)
                + ". state has multiple bad transition, which is not allowed in this setup.\n") {}
    };

    class BadChances : public TransitionInputError {
    public:
        explicit BadChances(const std::string& state, double value)
            : TransitionInputError("Sum of transition chances of state " + state
                                   + " is not 1 it is " + std::to_string(value) + ".\n") {}
    };

    class MissingStateName : public TransitionInputError {
    public:
        explicit MissingStateName(const std::string& name)
            : TransitionInputError(
                "State called " + name
                + " is missing from the file, but it should be there according to the logic.\n") {}
    };
}// namespace IOProgression

namespace IOParameters {
    class ParametersInputError : public CustomErrors {
    protected:
        explicit ParametersInputError(std::string&& error_p)
            : CustomErrors("Parameters input file error: " + error_p) {}
    };

    class NotBinary : public ParametersInputError {
    public:
        NotBinary() : ParametersInputError("There are two genders!!!\n") {}
    };

    class WrongGenderName : public ParametersInputError {
    public:
        WrongGenderName() : ParametersInputError("Define an F and an M genders.\n") {}
    };

    class NegativeInterval : public ParametersInputError {
    public:
        NegativeInterval(unsigned from, unsigned to)
            : ParametersInputError(
                "In age intervals, the from values should be smaller than to "
                "values ("
                + std::to_string(from) + ", " + std::to_string(to) + ").\n") {}
    };

    class NegativeFrom : public ParametersInputError {
    public:
        NegativeFrom()
            : ParametersInputError("From value in the age scaling should be positive value!\n") {}
    };
}// namespace IOParameters

namespace IOAgents {
    class AgentsInputError : public CustomErrors {
    protected:
        explicit AgentsInputError(std::string&& error_p)
            : CustomErrors("Agents input file error: " + error_p) {}
    };

    class InvalidGender : public AgentsInputError {
    public:
        explicit InvalidGender(const std::string& genderName)
            : AgentsInputError("Wrong gender name (" + genderName + ").\n") {}
    };

    class NotDefinedAge : public AgentsInputError {
    public:
        explicit NotDefinedAge(unsigned age)
            : AgentsInputError(
                "Age " + std::to_string(age) + " was not defined in the parameter input file!\n") {}
    };

    class NotDefinedCondition : public AgentsInputError {
    public:
        explicit NotDefinedCondition(std::string conditionID)
            : AgentsInputError("Condition with ID " + conditionID
                               + " was not defined in the parameter input file!\n") {}
    };

    class InvalidPPState : public AgentsInputError {
    public:
        explicit InvalidPPState(const std::string& PP)
            : AgentsInputError(PP + " is not a valid PP state.\n") {}
    };

    class InvalidAgentType : public AgentsInputError {
    public:
        explicit InvalidAgentType(unsigned ID)
            : AgentsInputError("Agent type ID " + std::to_string(ID)
                               + " does not exists in AgentTypes input file.\n") {}
    };

    class InvalidLocationID : public AgentsInputError {
    public:
        explicit InvalidLocationID(const std::string& ID)
            : AgentsInputError(
                "Location ID " + ID + " does not exists in Locations input file.\n") {}
    };

    class UnnecessaryLocType : public AgentsInputError {
    public:
        UnnecessaryLocType(unsigned agentID, unsigned aTypeID, unsigned lTypeID)
            : AgentsInputError("Agent with the index of " + std::to_string(agentID)
                               + " with the agent type ID of " + std::to_string(aTypeID)
                               + " does not need a location type with the ID of "
                               + std::to_string(lTypeID) + ".\n") {}
    };

    class MissingLocationType : public AgentsInputError {
    public:
        MissingLocationType(unsigned agentID, unsigned agentType, std::string&& missingTypes)
            : AgentsInputError("Agent with the index of " + std::to_string(agentID)
                               + " type "+std::to_string(agentType)+" does not have the following location types: " + missingTypes
                               + ".\n") {}
    };
}// namespace IOAgents

namespace IOLocations {
    class LocationsInputError : public CustomErrors {
    protected:
        explicit LocationsInputError(std::string&& error_p)
            : CustomErrors("Locations input file error: " + error_p) {}
    };

    class WrongState : public LocationsInputError {
    public:
        explicit WrongState(const std::string& state)
            : LocationsInputError("Wrong state name (" + state + ").\n") {}
    };
}// namespace IOLocations

namespace IOAgentTypes {
    class AgentTypesInputError : public CustomErrors {
    protected:
        explicit AgentTypesInputError(std::string&& error_p)
            : CustomErrors("Locations input file error: " + error_p) {}
    };

    class BadIDCommonSchedules : AgentTypesInputError {
    public:
        explicit BadIDCommonSchedules(unsigned ID)
            : AgentTypesInputError("Common schedule with the ID of " + std::to_string(ID)
                                   + " is not the next step.\n") {}
    };

    class InvalidWBStateInSchedule : AgentTypesInputError {
    public:
        explicit InvalidWBStateInSchedule(const std::string& wb)
            : AgentTypesInputError(wb + " is not a valid well-being state.\n") {}
    };

    class InvalidDayInSchedule : AgentTypesInputError {
    public:
        explicit InvalidDayInSchedule(const std::string& day)
            : AgentTypesInputError(day + " is not a valid day.\n") {}
    };

    class RepetitiveEventType : AgentTypesInputError {
    public:
        explicit RepetitiveEventType(const std::string& agentType)
            : AgentTypesInputError("Repetittive state for a schedule.\n") {}
    };
}// namespace IOAgentTypes

namespace init {
    class ProgramInit : public CustomErrors {
    protected:
        explicit ProgramInit(std::string&& error_p)
            : CustomErrors("Error during program initialization: " + error_p) {}
    };

    class BadTimeStep : public ProgramInit {
    public:
        explicit BadTimeStep(unsigned timeStep)
            : ProgramInit(
                "Time step of " + std::to_string(timeStep)
                + "min is not good, because 24 hours (1440 min) is not divisible by it.\n") {}
    };

    class BadInputFile : public ProgramInit {
    public:
        explicit BadInputFile(const std::string& fileName)
            : ProgramInit(fileName + " does not exists or cannot be opened.\n") {}
    };
}// namespace init
