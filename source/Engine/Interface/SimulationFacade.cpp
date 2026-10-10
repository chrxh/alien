#include "SimulationFacade.h"

SimulationFacade _SimulationFacade::_instance;

SimulationFacade _SimulationFacade::get()
{
    return _instance;
}
