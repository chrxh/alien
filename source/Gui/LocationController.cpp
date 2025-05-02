#include "LocationController.h"

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/LocationHelper.h"

#include "SimulationParametersBaseWidgets.h"
#include "SimulationParametersSourceWidgets.h"
#include "SimulationParametersLayerWidgets.h"

void LocationController::addLocationWindow(int orderNumber, RealVector2D const& initialPos)
{
    LocationWindow window;
    LocationWidgets widgets;
    if (orderNumber == 0) {
        auto baseWidgets = std::make_shared<_SimulationParametersBaseWidgets>();
        baseWidgets->init(_simulationFacade);
        widgets = baseWidgets;
    } else {
        auto parameters = _simulationFacade->getSimulationParameters();
        auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
        if (locationType == LocationType::Layer) {
            auto layerWidgets = std::make_shared<_SimulationParametersLayerWidgets>();
            layerWidgets->init(_simulationFacade, orderNumber);
            widgets = layerWidgets;
        } else {
            auto sourceWidgets = std::make_shared<_SimulationParametersSourceWidgets>();
            sourceWidgets->init(_simulationFacade, orderNumber);
            widgets = sourceWidgets;
        }
    }

    window.init(widgets, initialPos);
    _locationWindows.emplace_back(std::move(window));
}

void LocationController::deleteLocationWindow(int orderNumber)
{
    std::vector<LocationWindow> newlocationWindows;
    newlocationWindows.reserve(_locationWindows.size());

    for (auto& locationWindow : _locationWindows) {
        if (locationWindow.getOrderNumber() != orderNumber) {
            newlocationWindows.emplace_back(std::move(locationWindow));
        }
    }
    _locationWindows.swap(newlocationWindows);
}

void LocationController::remapLocationIndices(std::map<int, int> const& newByOldOrderNumber)
{
    for (auto& locationWindow : _locationWindows) {
        locationWindow.setOrderNumber(newByOldOrderNumber.at(locationWindow.getOrderNumber()));
    }
}

void LocationController::init(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void LocationController::process()
{
    if (!_sessionId.has_value() || _sessionId.value() != _simulationFacade->getSessionId()) {
        _locationWindows.clear();
    }

    std::vector<LocationWindow> newlocationWindows;
    newlocationWindows.reserve(_locationWindows.size());

    for (auto& locationWindow: _locationWindows) {
        locationWindow.process();
        if (locationWindow.isOn()) {
            newlocationWindows.emplace_back(std::move(locationWindow));
        }
    }
    _locationWindows.swap(newlocationWindows);

    _sessionId = _simulationFacade->getSessionId();
}
