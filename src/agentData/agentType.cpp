#include "agentType.h"
#include "customExceptions.h"

AgentTypeList::Event::Event()
    : locationType(0), chance(-1.0), start(0.0), end(0.0), duration(0.0) {}

AgentTypeList::Event::Event(const parser::AgentTypes::Type::ScheduleUnique::Event& in)
    : locationType(in.type),
      chance(static_cast<float>(in.chance)),
      start(static_cast<float>(in.start)),
      end(static_cast<float>(in.end)),
      duration(static_cast<float>(in.duration)) {}
// (4+1(dead)) WB state, 9 agent types, 7 days
HD unsigned AgentTypeList::getOffsetIndex(unsigned ID, states::WBStates state, Days day) {
    return (7 * static_cast<unsigned>(state)) + static_cast<unsigned>(day) + ID * 28;
}

AgentTypeList::AgentTypeList(std::size_t n) : eventOffset((n * 28) + 1, 0), events() {}

void AgentTypeList::addSchedule(unsigned ID,
    std::pair<states::WBStates, Days> state,
    const std::vector<AgentTypeList::Event>& schedule) {
    auto idx = getOffsetIndex(ID, state.first, state.second);
    auto n = schedule.size();
    for (auto i = idx + 1; i < eventOffset.size(); ++i) { eventOffset[i] += n; }
    events.insert(events.begin() + eventOffset[idx], schedule.begin(), schedule.end());
}

void AgentTypeList::unsetStayHome(double probability, unsigned homeType) {
    thrust::host_vector<unsigned> hoffsets(this->eventOffset);
    thrust::host_vector<AgentTypeList::Event> hevents(this->events);

    //Go through each schedule one by one
    for (unsigned s = 0; s < hoffsets.size()-1; s++) {
        unsigned window_start = hoffsets[s];
        unsigned window_end = window_start;
        while (window_start < hoffsets[s+1]) {
            //see how many events in this window
            while (hevents[window_end].start == hevents[window_start].start && window_end < hevents.size())
                window_end++;
            //Increase probability of all non-home events
            for (unsigned i = window_start; i < window_end; i++)
                if (hevents[i].locationType != homeType) 
                    hevents[i].chance /= (1.0-probability);
            //Do home
            for (unsigned i = window_start; i < window_end; i++) {
                if (hevents[i].locationType == homeType) {
                    //Decrease probability
                    hevents[i].chance = (hevents[i].chance - probability)/(1.0-probability);
                    if (std::abs(hevents[i].chance)<1e-10) {
                        hevents.erase(hevents.begin()+i);
                        for (int i = s+1; i < hoffsets.size(); i++)
                            hoffsets[i]--;
                    }
                    break;
                }
            }
            window_start=window_end;
        }
    }

    //upload
    this->eventOffset = hoffsets;
    this->events = hevents;
}

void AgentTypeList::setStayHome(double probability, unsigned homeType) {
    thrust::host_vector<unsigned> hoffsets(this->eventOffset);
    thrust::host_vector<AgentTypeList::Event> hevents(this->events);

    //Go through each schedule one by one
    for (unsigned s = 0; s < hoffsets.size()-1; s++) {
        unsigned window_start = hoffsets[s];
        unsigned window_end = window_start;
        while (window_start < hoffsets[s+1]) {
            //see how many events in this window
            while (hevents[window_end].start == hevents[window_start].start && window_end < hevents.size())
                window_end++;
            //check if home is in list
            bool home_present = false;
            for (unsigned i = window_start; i < window_end; i++)
                if (hevents[i].locationType == homeType) {
                    home_present = true;
                    //Increase probability by all others decreased
                    hevents[i].chance = hevents[i].chance + (1.0-hevents[i].chance)*probability;
                    break;
                }
            //if home in list, and only one, step to next window
            if (home_present && window_end-window_start == 1) {window_start = window_end; continue;}
            //if no home in list, insert event
            if (!home_present) {
                AgentTypeList::Event homeEvent;
                homeEvent.start = hevents[window_start].start;
                homeEvent.end = hevents[window_start].end;
                homeEvent.locationType = homeType;
                homeEvent.chance = probability;
                homeEvent.duration = hevents[window_start].duration;  //TODO: what should this be?
                hevents.insert(hevents.begin()+window_start, homeEvent);
                window_end++;
                for (int i = s+1; i < hoffsets.size(); i++)
                    hoffsets[i]++;
            }
            //Decrease probability of all other events
            for (unsigned i = window_start; i < window_end; i++)
                if (hevents[i].locationType != homeType) 
                    hevents[i].chance *= (1.0-probability);
            window_start=window_end;
        }
    }

    //upload
    this->eventOffset = hoffsets;
    this->events = hevents;
}