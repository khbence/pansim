#include "timing.h"

std::map<std::string, Timing::LoopData> Timing::loops;
std::vector<int> Timing::stack;
int Timing::counter = 0;

void Timing::reportWithParent(int parent, const std::string& indentation) {
    for (const auto& element : loops) {
        const LoopData& l = element.second;
        if (l.parent == parent) {
            std::cout << indentation + element.first + ": " + std::to_string(l.time) + " seconds\n";
            reportWithParent(l.index, indentation + "  ");
        }
    }
}

void Timing::startTimer(const std::string& _name) {
    auto now = std::chrono::system_clock::now();
    if (loops.size() == 0) counter = 0;
    int parent = stack.size() == 0 ? -1 : stack.back();
    std::string fullname = _name + "(" + std::to_string(parent) + ")";
    int index;
    if (loops.find(fullname) != loops.end()) {
        loops[fullname].current = now;
        index = loops[fullname].index;
    } else {
        loops[fullname] = { counter++, parent, 0.0, now };
        index = counter - 1;
    }
    stack.push_back(index);
}

void Timing::stopTimer(const std::string& _name) {
    stack.pop_back();
    int parent = stack.empty() ? -1 : stack.back();
    std::string fullname = _name + "(" + std::to_string(parent) + ")";
    auto now = std::chrono::system_clock::now();
    loops[fullname].time += std::chrono::duration<double>(now - loops[fullname].current).count();
}

void Timing::report() {
    std::vector<int> loopstack;
    int parent = -1;
    std::string indentation = "  ";
    for (const auto& element : loops) {
        const LoopData& l = element.second;
        if (l.parent == parent) {
            std::cout << indentation + element.first + ": " + std::to_string(l.time) + " seconds\n";
            reportWithParent(l.index, indentation + "  ");
        }
    }
}