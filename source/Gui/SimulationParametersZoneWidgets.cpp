#include "SimulationParametersZoneWidgets.h"

#include "EngineInterface/LocationHelper.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersValidationService.h"

#include "AlienImGui.h"
#include "SpecificationGuiService.h"
#include "SimulationInteractionController.h"

void _SimulationParametersZoneWidgets::init(SimulationFacade const& simulationFacade, int locationIndex)
{
    _simulationFacade = simulationFacade;
    _locationIndex = locationIndex;
}

void _SimulationParametersZoneWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto zoneIndex = LocationHelper::findLocationArrayIndex(parameters, _locationIndex);
    _zoneName = std::string(parameters.zoneName.zoneValues[zoneIndex]);

    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _simulationFacade, _locationIndex);

    if (parameters != lastParameters) {
        auto isRunning = _simulationFacade->isSimulationRunning();
        _simulationFacade->setSimulationParameters(
            parameters, isRunning ? SimulationParametersUpdateConfig::AllExceptChangingPositions : SimulationParametersUpdateConfig::All);
    }
}

std::string _SimulationParametersZoneWidgets::getLocationName()
{
    return "Simulation parameters for '" + _zoneName + "'";
}

int _SimulationParametersZoneWidgets::getLocationIndex() const
{
    return _locationIndex;
}

void _SimulationParametersZoneWidgets::setLocationIndex(int locationIndex)
{
    _locationIndex = locationIndex;
}
