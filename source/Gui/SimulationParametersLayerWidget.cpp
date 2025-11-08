#include "SimulationParametersLayerWidget.h"

#include <EngineInterface/LocationHelper.h>
#include <EngineInterface/ParametersValidationService.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "SimulationInteractionController.h"
#include "SpecificationGuiService.h"
#include <EngineInterface/SimulationFacade.h>

void _SimulationParameterLayerWidget::init(SimulationFacade const& simulationFacade, int orderNumber)
{

    _orderNumber = orderNumber;
}

void _SimulationParameterLayerWidget::process(ParametersFilter const& filter)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto layerIndex = LocationHelper::findLocationArrayIndex(parameters, _orderNumber);
    _layerName = std::string(parameters.layerName.layerValues[layerIndex]);

    ImGui::PushID("Layer");
    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _SimulationFacade::get(), _orderNumber, filter);
    ImGui::PopID();

    if (parameters != lastParameters) {
        ParametersValidationService::get().validateAndCorrect({_SimulationFacade::get()->getWorldSize()}, parameters);
        auto isRunning = _SimulationFacade::get()->isSimulationRunning();
        _SimulationFacade::get()->setSimulationParameters(
            parameters, isRunning ? SimulationParametersUpdateConfig::AllExceptChangingPositions : SimulationParametersUpdateConfig::All);
    }
}

std::string _SimulationParameterLayerWidget::getLocationName()
{
    return "Simulation parameters for '" + _layerName + "'";
}

int _SimulationParameterLayerWidget::getOrderNumber() const
{
    return _orderNumber;
}

void _SimulationParameterLayerWidget::setOrderNumber(int orderNumber)
{
    _orderNumber = orderNumber;
}
