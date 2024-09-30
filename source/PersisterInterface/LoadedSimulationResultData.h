#pragma once

#include <string>

#include "EngineInterface/DeserializedSimulation.h"

struct LoadedSimulationResultData
{
    std::string simulationName;
    DeserializedSimulation deserializedSimulation;
};
