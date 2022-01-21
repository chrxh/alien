#pragma once

#include <string>

#include "DllExport.h"

class StringHelper
{
public:
    BASE_EXPORT static std::string format(uint64_t n);
    BASE_EXPORT static std::string format(float v, int decimals);

    BASE_EXPORT static void copy(char* target, int targetSize, std::string const& source);
};