#pragma once

#include <optional>

#include "EngineInterface/SerializerService.h"

class SerializationHelperService
{
public:
    static DeserializedSimulation getDeserializedSerialization(SimulationController const& simController);
};
