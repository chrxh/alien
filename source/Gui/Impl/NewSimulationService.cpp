#include "NewSimulationService.h"

#include <Base/Interface/StringHelper.h>

#include <Data/Interface/SimulationParameters.h>

#include <Engine/Interface/SimulationFacade.h>
#include <Engine/Interface/TemporalControlService.h>

#include <Persister/Interface/PersisterFacade.h>

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
    TemporalControlService::get().createFlashback();
}

std::optional<std::string> NewSimulationService::loadSimulation(SimulationDesc const& simulation)
{
    _PersisterFacade::get()->shutdown();
    _SimulationFacade::get()->closeSimulation();

    std::optional<std::string> errorMessage;
    try {
        _SimulationFacade::get()->newSimulation(simulation._timestep, simulation._worldSize, simulation._simulationParameters);
        _SimulationFacade::get()->setRealTime(simulation._realTime);
        _SimulationFacade::get()->setSimulationData(simulation._mainData);
        _SimulationFacade::get()->setStatisticsHistory(simulation._statistics);
    } catch (std::exception const& exception) {
        errorMessage = exception.what();
    } catch (...) {
        errorMessage = "Failed to load simulation.";
    }

    if (errorMessage) {
        _SimulationFacade::get()->closeSimulation();
        _SimulationFacade::get()->newSimulation(simulation._timestep, simulation._worldSize, simulation._simulationParameters);
    }
    _PersisterFacade::get()->restart();

    Viewport::get().setCenterInWorldPos(simulation._center);
    Viewport::get().setZoomFactor(simulation._zoom);
    TemporalControlService::get().createFlashback();
    return errorMessage;
}
