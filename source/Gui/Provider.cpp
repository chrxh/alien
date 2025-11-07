#include "Provider.h"

SimulationFacade Provider::_simulationFacade;
PersisterFacade Provider::_persisterFacade;

void Provider::setSimulationFacade(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;
}

SimulationFacade Provider::getSimulationFacade()
{
    return _simulationFacade;
}

void Provider::setPersisterFacade(PersisterFacade const& persisterFacade)
{
    _persisterFacade = persisterFacade;
}

PersisterFacade Provider::getPersisterFacade()
{
    return _persisterFacade;
}
