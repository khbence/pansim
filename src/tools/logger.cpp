#include "logger.hpp"

const std::array<std::string, 4> Logger::levelNames = { "DEBUG", "INFO", "WARNING", "ERROR" };
LogLevel Logger::lowestLevel = LogLevel::DEBUG;

void Logger::setLowestLevel(LogLevel level) {
    lowestLevel = level;
}
