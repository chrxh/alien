#include "SimulationFacadeProvider.h"

SimulationFacade SimulationFacadeProvider::_simulationFacade;

void SimulationFacadeProvider::setSimulationFacade(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;
}

SimulationFacade SimulationFacadeProvider::getSimulationFacade()
{
    return _simulationFacade;
}
