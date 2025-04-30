#pragma once

#include <chrono>
#include <string>

#include "Base/Definitions.h"

class StringHelper
{
public:
    static std::string format(uint64_t n, char separator = ',');
    static std::string format(float v, int decimalsAfterPoint);
    static std::string format(std::chrono::seconds duration);
    static std::string format(std::chrono::milliseconds duration);
    static std::string format(std::chrono::system_clock::time_point const& timePoint);

    static void copy(char* target, int maxSize, std::string const& source);
    static bool compare(char const* target, int maxSize, char const* source);

    static bool containsCaseInsensitive(std::string const& str, std::string const& toMatch);

    struct Decomposition
    {
        std::string beforeMatch;
        std::string match;
    };
    static Decomposition decomposeCaseInsensitiveMatch(std::string const& str, std::string const& toMatch);
};