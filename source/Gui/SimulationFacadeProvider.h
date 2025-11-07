#pragma once

#include <EngineInterface/SimulationFacade.h>

class SimulationFacadeProvider
{
public:
    static void setSimulationFacade(SimulationFacade const& simulationFacade);
    static SimulationFacade getSimulationFacade();

private:
    static SimulationFacade _simulationFacade;
};
