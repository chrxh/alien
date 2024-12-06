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
    std::vector<LocationWindow> newlocationWindows;
    newlocationWindows.reserve(_locationWindows.size());

    for (auto& locationWindow: _locationWindows) {
        locationWindow.process();
        if (locationWindow.isOn()) {
            newlocationWindows.emplace_back(std::move(locationWindow));
        }
    }
    _locationWindows.swap(newlocationWindows);
}
