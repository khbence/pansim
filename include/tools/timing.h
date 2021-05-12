#pragma once
#include <chrono>
#include <map>
#include <vector>
#include <iostream>

#define PROFILING 1

class Timing {
    struct LoopData {
        int index = 0;
        int parent = 0;
        double time = 0.0;
        std::chrono::time_point<std::chrono::system_clock> current;
    };
    std::string name;

    static std::map<std::string, LoopData> loops;
    static std::vector<int> stack;
    static int counter;

    static void reportWithParent(int parent, const std::string& indentation);

public:
    explicit Timing(std::string&& name_p) : name(std::move(name_p)) { startTimer(name); }
    ~Timing() { stopTimer(name); }

    static void startTimer(const std::string& _name);
    static void stopTimer(const std::string& _name);
    static void report();
};

#if PROFILING
#define BEGIN_PROFILING(name) Timing::startTimer(name);
#define END_PROFILING(name) Timing::stopTimer(name);

#define PROFILE_SCOPE(name) Timing timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else
#define BEGIN_PROFILING(name)
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#define END_PROFILING(name)
#endif
