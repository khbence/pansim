#pragma once
#include <string>
#include <array>
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "fmt/chrono.h"

enum class LogLevel { None = 0, DEBUG, INFO, WARNING, ERROR };

class Logger {
    static const std::array<std::string, 4> levelNames;
    static LogLevel lowestLevel;

    template<LogLevel LEVEL, typename... T>
    static void log(fmt::format_string<T...> msg, T&&... args) {
        if (LEVEL >= lowestLevel) {
            auto rawMessage = fmt::format(std::move(msg), std::forward<T>(args)...);
            auto time =
                fmt::format("{:%Y-%m-%d %H:%M:%S}", std::chrono::high_resolution_clock::now());
            const std::string delimiter = "\n";
            typename decltype(rawMessage)::size_type start = 0;
            auto end = rawMessage.find(delimiter);
            do {
                auto part = rawMessage.substr(start, end - start);
                fmt::print("[{}] {}: {}\n",
                    levelNames.at(static_cast<std::size_t>(LEVEL) - 1),
                    time,
                    part);
                start = end + delimiter.length();
                end = rawMessage.find(delimiter, start);
            } while (end != std::string::npos);
        }
    }

public:
    static void setLowestLevel(LogLevel level);

    template<typename... T>
    static void debug(fmt::format_string<T...> msg, T&&... args) {
        log<LogLevel::DEBUG>(msg, args...);
    }

    template<typename... T>
    static void info(fmt::format_string<T...> msg, T&&... args) {
        log<LogLevel::INFO>(msg, args...);
    }

    template<typename... T>
    static void warning(fmt::format_string<T...> msg, T&&... args) {
        log<LogLevel::WARNING>(msg, args...);
    }

    template<typename... T>
    static void error(fmt::format_string<T...> msg, T&&... args) {
        log<LogLevel::ERROR>(msg, args...);
    }
};