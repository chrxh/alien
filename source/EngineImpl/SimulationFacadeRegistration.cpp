#include "SimulationFacadeRegistration.h"

#include "SimulationFacadeImpl.h"

SimulationFacadeRegistration::SimulationFacadeRegistration()
{
    _SimulationFacade::set(std::make_shared<_SimulationFacadeImpl>());
}

// Global variable to trigger registration
static SimulationFacadeRegistration simulationFacadeRegistration;
