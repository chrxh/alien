#include "SimulationParametersLayerWidget.h"

#include <EngineInterface/LocationHelper.h>
#include <EngineInterface/ParametersValidationService.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "SimulationInteractionController.h"
#include "SpecificationGuiService.h"
#include "SimulationFacadeProvider.h"

void _SimulationParameterLayerWidget::init(SimulationFacade const& simulationFacade, int orderNumber)
{

    _orderNumber = orderNumber;
}

void _SimulationParameterLayerWidget::process(ParametersFilter const& filter)
{
    auto parameters = SimulationFacadeProvider::getSimulationFacade()->getSimulationParameters();
    auto origParameters = SimulationFacadeProvider::getSimulationFacade()->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto layerIndex = LocationHelper::findLocationArrayIndex(parameters, _orderNumber);
    _layerName = std::string(parameters.layerName.layerValues[layerIndex]);

    ImGui::PushID("Layer");
    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, SimulationFacadeProvider::getSimulationFacade(), _orderNumber, filter);
    ImGui::PopID();

    if (parameters != lastParameters) {
        ParametersValidationService::get().validateAndCorrect({SimulationFacadeProvider::getSimulationFacade()->getWorldSize()}, parameters);
        auto isRunning = SimulationFacadeProvider::getSimulationFacade()->isSimulationRunning();
        SimulationFacadeProvider::getSimulationFacade()->setSimulationParameters(
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
