#include "NewSimulationService.h"

#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParameters.h>

#include "TemporalControlWindow.h"
#include "Viewport.h"

namespace
{
    auto constexpr ProjectNameSize = sizeof(Char64) / sizeof(char);
    auto constexpr InitialZoomFactor = 4.0f;
}

void NewSimulationService::createSimulation(Parameters const& parameters)
{
    SimulationParameters simulationParameters;
    if (parameters._adoptSimulationParameters) {
        simulationParameters = _SimulationFacade::get()->getSimulationParameters();
    }
    StringHelper::copy(simulationParameters.projectName.value, ProjectNameSize, parameters._projectName);
    simulationParameters.externalEnergy.value = parameters._externalEnergy;
    _SimulationFacade::get()->closeSimulation();

    _SimulationFacade::get()->newSimulation(0, parameters._worldSize, simulationParameters);
    Viewport::get().setCenterInWorldPos({toFloat(parameters._worldSize.x) / 2, toFloat(parameters._worldSize.y) / 2});
    Viewport::get().setZoomFactor(InitialZoomFactor);
    TemporalControlWindow::get().onSnapshot();
}
