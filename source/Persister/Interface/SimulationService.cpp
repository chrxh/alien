#include "SimulationService.h"

#include <Base/StringHelper.h>

#include <Data/SimulationParameters.h>

#include <Engine/Interface/SimulationFacade.h>
#include <Engine/Interface/TemporalControlService.h>

#include "PersisterFacade.h"

namespace
{
    auto constexpr ProjectNameSize = sizeof(Char64) / sizeof(char);
}

void SimulationService::createSimulation(NewSimulationParameters const& parameters)
{
    SimulationParameters simulationParameters;
    if (parameters._adoptSimulationParameters) {
        simulationParameters = _SimulationFacade::get()->getSimulationParameters();
    }
    StringHelper::copy(simulationParameters.projectName.value, ProjectNameSize, parameters._projectName);
    simulationParameters.externalEnergy.value = parameters._externalEnergy;
    _SimulationFacade::get()->closeSimulation();

    _SimulationFacade::get()->newSimulation(0, parameters._worldSize, simulationParameters);
    TemporalControlService::get().createFlashback();
}

std::optional<std::string> SimulationService::loadSimulation(SimulationDesc simulation)
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

    if (errorMessage) {
        TemporalControlService::get().createFlashback();
    } else {
        TemporalControlService::get().createFlashback(std::move(simulation._mainData));
    }
    return errorMessage;
}
