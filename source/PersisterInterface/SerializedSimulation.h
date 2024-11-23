#pragma once

#include "Definitions.h"

struct SerializedSimulation
{
    std::string mainData;       //binary
    std::string auxiliaryData;  //JSON
    std::string statistics;     //CSV
};
