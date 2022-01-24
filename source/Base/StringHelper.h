#pragma once

#include <string>

class StringHelper
{
public:
    static std::string format(uint64_t n);
    static std::string format(float v, int decimals);

    static void copy(char* target, int targetSize, std::string const& source);
};