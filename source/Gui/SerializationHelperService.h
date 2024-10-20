#pragma once

#include "Base/Singleton.h"
#include "PersisterInterface/DeserializedSimulation.h"
#include "EngineInterface/SimulationFacade.h"

class SerializationHelperService
{
    MAKE_SINGLETON(SerializationHelperService);

public:
    DeserializedSimulation getDeserializedSerialization(SimulationFacade const& simulationFacade);
};
