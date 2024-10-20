#pragma once

#include <string>

#include "PersisterInterface/DeserializedSimulation.h"

struct ReadSimulationResultData
{
    std::string simulationName;
    DeserializedSimulation deserializedSimulation;
};
