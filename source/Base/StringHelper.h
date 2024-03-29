#pragma once

#include <string>

#include "Base/Definitions.h"

class StringHelper
{
public:
    static std::string format(uint64_t n);
    static std::string format(float v, int decimalsAfterPoint);

    static void copy(char* target, int targetSize, std::string const& source);
};