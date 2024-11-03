#pragma once

#include <filesystem>

#include "SharedDeserializedSimulation.h"

struct SaveDeserializedSimulationRequestData
{
    std::filesystem::path filename;
    SharedDeserializedSimulation sharedDeserializedSimulation;
    bool generateNameFromTimestep = false;
};
