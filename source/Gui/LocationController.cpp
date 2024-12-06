#include "LocationController.h"

#include "EngineInterface/SimulationFacade.h"

#include "LocationHelper.h"
#include "SimulationParametersBaseWidgets.h"
#include "SimulationParametersZoneWidgets.h"

void LocationController::addLocationWindow(int locationIndex)
{
    LocationWindow window;

    if (locationIndex == 0) {
        auto widgets = std::make_shared<_SimulationParametersBaseWidgets>();
        widgets->init(_simulationFacade);
        window.init("Base parameters", widgets);
    } else {
        auto parameters = _simulationFacade->getSimulationParameters();
        auto location = LocationHelper::findLocation(parameters, locationIndex);
        if (std::holds_alternative<SimulationParametersZone*>(location)) {
            auto zoneIndex = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            auto widgets = std::make_shared<_SimulationParametersZoneWidgets>();
            widgets->init(_simulationFacade, zoneIndex);
            window.init("Zone parameters", widgets);
        } else {
            
        }
    }

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
