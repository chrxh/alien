#pragma once

#include "Base/Singleton.h"

#include "SimulationParametersDescription.h"

class SimulationParametersDescriptionService
{
    MAKE_SINGLETON(SimulationParametersDescriptionService);

public:
    SimulationParametersDescription createSimulationParametersDescription() const;
};
