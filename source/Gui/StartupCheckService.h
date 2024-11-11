#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationFacade.h"

class StartupCheckService
{
    MAKE_SINGLETON(StartupCheckService);

public:
    void check(SimulationFacade const& simulationFacade);
};
