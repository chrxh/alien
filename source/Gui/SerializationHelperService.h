#pragma once

#include "EngineInterface/DeserializedSimulation.h"
#include "EngineInterface/SimulationController.h"

class SerializationHelperService
{
public:
    static DeserializedSimulation getDeserializedSerialization(SimulationController const& simController);
};
