#include "SimulationParametersSourceWidget.h"

#include <imgui.h>

#include <EngineInterface/LocationHelper.h>
#include <EngineInterface/ParametersValidationService.h>
#include <EngineInterface/SimulationFacade.h>

#include "SimulationInteractionController.h"
#include "SpecificationGuiService.h"
#include "Provider.h"

void _SimulationParametersSourceWidgets::init(SimulationFacade const& simulationFacade, int orderNumber)
{

    _orderNumber = orderNumber;
}

void _SimulationParametersSourceWidgets::process(ParametersFilter const& filter)
{
    auto parameters = Provider::getSimulationFacade()->getSimulationParameters();
    auto origParameters = Provider::getSimulationFacade()->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto sourceIndex = LocationHelper::findLocationArrayIndex(parameters, _orderNumber);

    _sourceName = std::string(parameters.sourceName.sourceValues[sourceIndex]);

    ImGui::PushID("Source");
    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, Provider::getSimulationFacade(), _orderNumber, filter);
    ImGui::PopID();

    if (parameters != lastParameters) {
        ParametersValidationService::get().validateAndCorrect({Provider::getSimulationFacade()->getWorldSize()}, parameters);
        auto isRunning = Provider::getSimulationFacade()->isSimulationRunning();
        Provider::getSimulationFacade()->setSimulationParameters(
            parameters, isRunning ? SimulationParametersUpdateConfig::AllExceptChangingPositions : SimulationParametersUpdateConfig::All);
    }
}

std::string _SimulationParametersSourceWidgets::getLocationName()
{
    return "Simulation parameters for '" + _sourceName + "'";
}

int _SimulationParametersSourceWidgets::getOrderNumber() const
{
    return _orderNumber;
}

void _SimulationParametersSourceWidgets::setOrderNumber(int orderNumber)
{
    _orderNumber = orderNumber;
}
