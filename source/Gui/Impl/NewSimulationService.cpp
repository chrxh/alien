#include "NewSimulationService.h"

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <Data/SimulationParameters.h>

#include <Engine/Interface/SimulationFacade.h>
#include <Engine/Interface/TemporalControlService.h>

#include <Network/NetworkService.h>

#include <Persister/Interface/PersisterFacade.h>

#include "Viewport.h"

namespace
{
    auto constexpr ProjectNameSize = sizeof(Char64) / sizeof(char);
    auto constexpr InitialZoomFactor = 4.0f;
    auto constexpr AutosaveOriginSettingsKey = "services.new simulation.autosave origin";
    auto constexpr AutosaveOriginServerSettingsKey = "services.new simulation.autosave origin server";
    auto constexpr AutosaveNameInLibrarySettingsKey = "services.new simulation.autosave name in library";
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
    _originResourceId.reset();
    _nameInLibrary.reset();
}

std::optional<std::string> NewSimulationService::loadSimulation(
    SimulationDesc const& simulation,
    std::optional<std::string> const& originResourceId,
    std::optional<std::string> const& nameInLibrary)
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
    _originResourceId = errorMessage ? std::nullopt : originResourceId;
    _nameInLibrary = errorMessage ? std::nullopt : nameInLibrary;
    return errorMessage;
}

std::optional<std::string> const& NewSimulationService::getOriginResourceId() const
{
    return _originResourceId;
}

void NewSimulationService::setOriginResourceId(std::optional<std::string> const& value)
{
    _originResourceId = value;
}

std::optional<std::string> const& NewSimulationService::getNameInLibrary() const
{
    return _nameInLibrary;
}

void NewSimulationService::setNameInLibrary(std::optional<std::string> const& value)
{
    _nameInLibrary = value;
}

void NewSimulationService::storeAutosaveOrigin() const
{
    GlobalSettings::get().setValue(AutosaveOriginSettingsKey, _originResourceId.value_or(std::string()));
    GlobalSettings::get().setValue(AutosaveOriginServerSettingsKey, NetworkService::get().getServerAddress());
    GlobalSettings::get().setValue(AutosaveNameInLibrarySettingsKey, _nameInLibrary.value_or(std::string()));
}

void NewSimulationService::restoreAutosaveOrigin()
{
    auto originResourceId = GlobalSettings::get().getValue(AutosaveOriginSettingsKey, std::string());
    auto originServer = GlobalSettings::get().getValue(AutosaveOriginServerSettingsKey, std::string());
    if (!originResourceId.empty() && originServer == NetworkService::get().getServerAddress()) {
        _originResourceId = originResourceId;
    }
    auto nameInLibrary = GlobalSettings::get().getValue(AutosaveNameInLibrarySettingsKey, std::string());
    if (!nameInLibrary.empty()) {
        _nameInLibrary = nameInLibrary;
    }
}
