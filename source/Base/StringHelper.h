#pragma once

#include <chrono>
#include <string>

#include "Base/Definitions.h"

class StringHelper
{
public:
    static std::string format(uint64_t n, char separator = ',');
    static std::string format(float v, int decimalsAfterPoint);
    static std::string format(std::chrono::milliseconds duration);
    static std::string format(std::chrono::system_clock::time_point const& timePoint);

    static void copy(char* target, int targetSize, std::string const& source);
};