#pragma once

#include <Base/Singleton.h>

#include <Engine/Interface/SimulationFacade.h>

class StartupCheckService
{
    MAKE_SINGLETON(StartupCheckService);

public:
    void check();
};
