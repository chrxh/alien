#pragma once

#include <string>

#include "EngineInterface/DeserializedSimulation.h"

struct ReadSimulationResultData
{
    std::string simulationName;
    DeserializedSimulation deserializedSimulation;
};
