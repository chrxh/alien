#include "LocationController.h"

#include <EngineInterface/LocationHelper.h>
#include <EngineInterface/SimulationFacade.h>

#include "SimulationParametersBaseWidget.h"
#include "SimulationParametersLayerWidget.h"
#include "SimulationParametersSourceWidget.h"

void LocationController::addLocationWindow(int orderNumber, RealVector2D const& initialPos)
{
    LocationWindow window;
    LocationWidget widget;
    if (orderNumber == 0) {
        auto baseWidgets = std::make_shared<_SimulationParametersBaseWidget>();
        baseWidgets->init();
        widget = baseWidgets;
    } else {
        auto parameters = _SimulationFacade::get()->getSimulationParameters();
        auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
        if (locationType == LocationType::Layer) {
            auto layerWidgets = std::make_shared<_SimulationParameterLayerWidget>();
            layerWidgets->init(orderNumber);
            widget = layerWidgets;
        } else {
            auto sourceWidgets = std::make_shared<_SimulationParametersSourceWidgets>();
            sourceWidgets->init(orderNumber);
            widget = sourceWidgets;
        }
    }

    window.init(widget, initialPos);
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

void LocationController::init() {}

void LocationController::process()
{
    if (!_sessionId.has_value() || _sessionId.value() != _SimulationFacade::get()->getSessionId()) {
        _locationWindows.clear();
    }

    std::vector<LocationWindow> newlocationWindows;
    newlocationWindows.reserve(_locationWindows.size());

    for (auto& locationWindow : _locationWindows) {
        locationWindow.process();
        if (locationWindow.isOn()) {
            newlocationWindows.emplace_back(std::move(locationWindow));
        }
    }
    _locationWindows.swap(newlocationWindows);

    _sessionId = _SimulationFacade::get()->getSessionId();
}
