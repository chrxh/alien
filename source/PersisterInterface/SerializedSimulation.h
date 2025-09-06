#pragma once

#include "Definitions.h"

struct SerializedSimulation
{
    std::string mainData;       // Binary
    std::string auxiliaryData;  // JSON
    std::string statistics;     // CSV
};
