#pragma once

#include <EngineInterface/SimulationFacade.h>
#include <PersisterInterface/PersisterFacade.h>

class Provider
{
public:
    static void setSimulationFacade(SimulationFacade const& simulationFacade);
    static SimulationFacade getSimulationFacade();

    static void setPersisterFacade(PersisterFacade const& persisterFacade);
    static PersisterFacade getPersisterFacade();

private:
    static SimulationFacade _simulationFacade;
    static PersisterFacade _persisterFacade;
};
