#include "LocationController.h"

#include "SimulationParametersBaseWidgets.h"
#include "EngineInterface/SimulationFacade.h"

void LocationController::addBase()
{
    auto widgets = std::make_shared<_SimulationParametersBaseWidgets>();
    widgets->init(_simulationFacade);

    LocationWindow window;
    window.init("Base parameters", widgets);
    _locationWindows.emplace_back(std::move(window));
}

void LocationController::init(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void LocationController::process()
{
    for (auto& locationWindow: _locationWindows) {
        locationWindow.process();
    }
}
