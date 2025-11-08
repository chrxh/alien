#include "SimulationFacade.h"

SimulationFacade _SimulationFacade::_instance;

void _SimulationFacade::set(SimulationFacade const& instance)
{
    _instance = instance;
}

SimulationFacade _SimulationFacade::get()
{
    return _instance;
}
