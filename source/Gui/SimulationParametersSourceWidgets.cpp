#include "SimulationParametersSourceWidgets.h"

#include <imgui.h>

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersValidationService.h"
#include "EngineInterface/LocationHelper.h"

#include "SimulationInteractionController.h"
#include "SpecificationGuiService.h"

void _SimulationParametersSourceWidgets::init(SimulationFacade const& simulationFacade, int locationIndex)
{
    _simulationFacade = simulationFacade;
    _locationIndex = locationIndex;
}

void _SimulationParametersSourceWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto sourceIndex = LocationHelper::findLocationArrayIndex(parameters, _locationIndex);

    _sourceName = std::string(parameters.sourceName.sourceValues[sourceIndex]);

    ImGui::PushID("Source");
    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _simulationFacade, _locationIndex);
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

int _SimulationParametersSourceWidgets::getLocationIndex() const
{
    return _locationIndex;
}

void _SimulationParametersSourceWidgets::setLocationIndex(int locationIndex)
{
    _locationIndex = locationIndex;
}
