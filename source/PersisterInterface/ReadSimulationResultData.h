#pragma once

#include <filesystem>

#include "PersisterInterface/DeserializedSimulation.h"

struct ReadSimulationResultData
{
    std::filesystem::path filename;
    DeserializedSimulation deserializedSimulation;
};
