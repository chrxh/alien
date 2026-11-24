#include "SimulationSessionService.h"

#include <Base/ExitScopeGuard.h>
#include <Base/StringHelper.h>

#include <Data/SimulationParameters.h>

#include <Engine/Interface/SimulationFacade.h>
#include <Engine/Interface/TemporalControlService.h>

#include "PersisterFacade.h"

namespace
{
    auto constexpr ProjectNameSize = sizeof(Char64) / sizeof(char);
}

void SimulationSessionService::createSimulation(NewSimulationParameters const& parameters)
{
    SimulationParameters simulationParameters;
    if (parameters._adoptSimulationParameters) {
        simulationParameters = _SimulationFacade::get()->getSimulationParameters();
    }
    StringHelper::copy(simulationParameters.projectName.value, ProjectNameSize, parameters._projectName);
    simulationParameters.externalEnergy.value = parameters._externalEnergy;

    replaceSimulation([&] { _SimulationFacade::get()->newSimulation(0, parameters._worldSize, simulationParameters); });
    TemporalControlService::get().createFlashback();
}

std::optional<std::string> SimulationSessionService::loadSimulation(SimulationDesc simulation)
{
    std::optional<std::string> errorMessage;
    replaceSimulation([&] {
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
    });

    if (errorMessage) {
        TemporalControlService::get().createFlashback();
    } else {
        TemporalControlService::get().createFlashback(std::move(simulation._mainData));
    }
    return errorMessage;
}

void SimulationSessionService::replaceSimulation(std::function<void()> const& setupFunc)
{
    _PersisterFacade::get()->shutdown();
    ExitScopeGuard restartPersister([] { _PersisterFacade::get()->restart(); });

    _SimulationFacade::get()->closeSimulation();
    setupFunc();
}
