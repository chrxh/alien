#pragma once

#include "EngineInterface/DeserializedSimulation.h"
#include "EngineInterface/SimulationFacade.h"

class SerializationHelperService
{
public:
    static DeserializedSimulation getDeserializedSerialization(SimulationFacade const& simulationFacade);
};
