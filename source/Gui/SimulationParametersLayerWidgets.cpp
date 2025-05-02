#include "SimulationParametersLayerWidgets.h"

#include "EngineInterface/LocationHelper.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersValidationService.h"

#include "AlienImGui.h"
#include "SpecificationGuiService.h"
#include "SimulationInteractionController.h"

void _SimulationParametersLayerWidgets::init(SimulationFacade const& simulationFacade, int orderNumber)
{
    _simulationFacade = simulationFacade;
    _orderNumber = orderNumber;
}

void _SimulationParametersLayerWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto layerIndex = LocationHelper::findLocationArrayIndex(parameters, _orderNumber);
    _layerName = std::string(parameters.layerName.layerValues[layerIndex]);

    ImGui::PushID("Layer");
    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _simulationFacade, _orderNumber);
    ImGui::PopID();

    if (parameters != lastParameters) {
        auto isRunning = _simulationFacade->isSimulationRunning();
        _simulationFacade->setSimulationParameters(
            parameters, isRunning ? SimulationParametersUpdateConfig::AllExceptChangingPositions : SimulationParametersUpdateConfig::All);
    }
}

std::string _SimulationParametersLayerWidgets::getLocationName()
{
    return "Simulation parameters for '" + _layerName + "'";
}

int _SimulationParametersLayerWidgets::getOrderNumber() const
{
    return _orderNumber;
}

void _SimulationParametersLayerWidgets::setOrderNumber(int orderNumber)
{
    _orderNumber = orderNumber;
}
