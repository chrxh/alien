#include "SimulationParametersSourceWidgets.h"

#include <imgui.h>

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersValidationService.h"
#include "EngineInterface/LocationHelper.h"

#include "SimulationInteractionController.h"
#include "SpecificationGuiService.h"

void _SimulationParametersSourceWidgets::init(SimulationFacade const& simulationFacade, int orderNumber)
{
    _simulationFacade = simulationFacade;
    _orderNumber = orderNumber;
}

void _SimulationParametersSourceWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto sourceIndex = LocationHelper::findLocationArrayIndex(parameters, _orderNumber);

    _sourceName = std::string(parameters.sourceName.sourceValues[sourceIndex]);

    ImGui::PushID("Source");
    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _simulationFacade, _orderNumber);
    ImGui::PopID();

    if (parameters != lastParameters) {
        auto isRunning = _simulationFacade->isSimulationRunning();
        _simulationFacade->setSimulationParameters(
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
