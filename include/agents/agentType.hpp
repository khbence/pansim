#pragma once
#include <string>
#include "agentsFormat.hpp"
#include <vector>
#include "globalStates.hpp"
#include "timeHandler.hpp"
#include "timeDay.hpp"
#include "unordered_map"
#include "datatypes.hpp"

class AgentTypeList {
public:
    class Event {
    public:
        unsigned locationType;
        float chance;
        TimeDay start;
        TimeDay end;
        TimeDayDuration duration;

    public:
        Event();
        explicit Event(const parser::AgentTypes::Type::ScheduleUnique::Event& in);
    };

public:
    // use the getOffsetIndex function to get the position for this vector, and
    // use this value to search in events
    thrust::device_vector<unsigned> eventOffset;
    thrust::device_vector<Event> events;

    [[nodiscard]] static HD unsigned getOffsetIndex(unsigned ID, states::WBStates state, Days day);

    AgentTypeList() = default;
    AgentTypeList(std::size_t n);
    void addSchedule(unsigned ID, std::pair<states::WBStates, Days> state, const std::vector<Event>& schedule);
    void unsetStayHome(double probability, unsigned homeType);
    void setStayHome(double probability, unsigned homeType);
};