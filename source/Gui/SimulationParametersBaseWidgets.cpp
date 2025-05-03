#include "SimulationParametersBaseWidgets.h"

#include <imgui.h>

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersEditService.h"
#include "EngineInterface/SimulationParametersUpdateConfig.h"
#include "EngineInterface/ParametersValidationService.h"

#include "AlienImGui.h"
#include "SpecificationGuiService.h"

void _SimulationParametersBaseWidgets::init(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void _SimulationParametersBaseWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _simulationFacade, 0);

    if (parameters != lastParameters) {
        ParametersValidationService::get().validateAndCorrect({_simulationFacade->getWorldSize()}, parameters);
        _simulationFacade->setSimulationParameters(parameters, SimulationParametersUpdateConfig::AllExceptChangingPositions);
    }
}

std::string _SimulationParametersBaseWidgets::getLocationName()
{
    return "Simulation parameters for 'Base'";
}

int _SimulationParametersBaseWidgets::getOrderNumber() const
{
    return 0;
}

void _SimulationParametersBaseWidgets::setOrderNumber(int orderNumber)
{
    // do nothing
}
