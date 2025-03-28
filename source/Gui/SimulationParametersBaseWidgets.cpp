#include "SimulationParametersBaseWidgets.h"

#include <imgui.h>

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SimulationParametersEditService.h"
#include "EngineInterface/SimulationParametersUpdateConfig.h"
#include "EngineInterface/SimulationParametersValidationService.h"
#include "EngineInterface/SimulationParametersSpecificationService.h"
#include "EngineInterface/CellTypeStrings.h"

#include "AlienImGui.h"
#include "ParametersSpecGuiService.h"
#include "SimulationParametersMainWindow.h"

void _SimulationParametersBaseWidgets::init(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;
    for (int i = 0; i < CellType_Count; ++i) {
        _cellTypeStrings.emplace_back(Const::CellTypeToStringMap.at(i));
    }
}

void _SimulationParametersBaseWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    ParametersSpecGuiService::get().createWidgetsForParameters(0, parameters, origParameters);

    SimulationParametersValidationService::get().validateAndCorrect(parameters);

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
