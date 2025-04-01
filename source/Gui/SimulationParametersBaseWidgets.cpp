#include "SimulationParametersBaseWidgets.h"

#include <imgui.h>

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersEditService.h"
#include "EngineInterface/SimulationParametersUpdateConfig.h"
#include "EngineInterface/ParametersValidationService.h"
#include "EngineInterface/SpecificationService.h"

#include "AlienImGui.h"
#include "ParametersSpecGuiService.h"

void _SimulationParametersBaseWidgets::init(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void _SimulationParametersBaseWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    ParametersSpecGuiService::get().createWidgetsForParameters(parameters, origParameters, 0);

    ParametersValidationService::get().validateAndCorrect(parameters);

    if (parameters != lastParameters) {
        _simulationFacade->setSimulationParameters(parameters, SimulationParametersUpdateConfig::AllExceptChangingPositions);
    }
}

std::string _SimulationParametersBaseWidgets::getLocationName()
{
    return "Simulation parameters for 'Base'";
}

int _SimulationParametersBaseWidgets::getLocationIndex() const
{
    return 0;
}

void _SimulationParametersBaseWidgets::setLocationIndex(int locationIndex)
{
    // do nothing
}
