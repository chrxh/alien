#pragma once

#include <string>

#include "DllExport.h"

class StringFormatter
{
public:
    BASE_EXPORT static std::string format(uint64_t n);
    BASE_EXPORT static std::string format(float v, int decimals);
};