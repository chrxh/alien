#include "SimulationParametersLayerWidgets.h"

#include "EngineInterface/LocationHelper.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersValidationService.h"

#include "AlienImGui.h"
#include "SpecificationGuiService.h"
#include "SimulationInteractionController.h"

void _SimulationParametersLayerWidgets::init(SimulationFacade const& simulationFacade, int locationIndex)
{
    _simulationFacade = simulationFacade;
    _locationIndex = locationIndex;
}

void _SimulationParametersLayerWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto layerIndex = LocationHelper::findLocationArrayIndex(parameters, _locationIndex);
    _layerName = std::string(parameters.layerName.layerValues[layerIndex]);

    ImGui::PushID("Layer");
    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _simulationFacade, _locationIndex);
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

int _SimulationParametersLayerWidgets::getLocationIndex() const
{
    return _locationIndex;
}

void _SimulationParametersLayerWidgets::setLocationIndex(int locationIndex)
{
    _locationIndex = locationIndex;
}
