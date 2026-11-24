#include "SimulationGuiService.h"

#include <functional>

#include <Base/GlobalSettings.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

#include <Network/NetworkService.h>

#include <Persister/Interface/LocalLibraryService.h>

#include "Viewport.h"

namespace
{
    auto constexpr InitialZoomFactor = 4.0f;
    auto constexpr ExitSaveOriginSettingsKey = "services.new simulation.exit save origin";
    auto constexpr OriginKeySuffix = ".origin";
    auto constexpr OriginServerKeySuffix = ".origin server";
    auto constexpr NameInLibraryKeySuffix = ".name in library";
    auto constexpr LocalFileKeySuffix = ".local file";

    // Every folder ALIEN runs in has an exit save of its own, whose origin other ALIEN instances must not overwrite in the shared settings
    std::string getExitSaveOriginSettingsKey()
    {
        auto exitSavePath = StringHelper::toUtf8(LocalLibraryService::get().normalize(Const::AutosaveFile));
        return std::string(ExitSaveOriginSettingsKey) + "." + StringHelper::formatInHex(std::hash<std::string>{}(exitSavePath));
    }
}

void SimulationGuiService::createSimulation(Parameters const& parameters)
{
    SimulationService::get().createSimulation(parameters);
    Viewport::get().setCenterInWorldPos({toFloat(parameters._worldSize.x) / 2, toFloat(parameters._worldSize.y) / 2});
    Viewport::get().setZoomFactor(InitialZoomFactor);
    _originResourceId.reset();
    _nameInLibrary.reset();
    _localFile.reset();
    ++_simulationRevision;
}

std::optional<std::string> SimulationGuiService::loadSimulation(
    SimulationDesc simulation,
    std::optional<std::string> const& originResourceId,
    std::optional<std::string> const& nameInLibrary,
    std::optional<std::filesystem::path> const& localFile)
{
    auto center = simulation._center;
    auto zoom = simulation._zoom;
    auto errorMessage = SimulationService::get().loadSimulation(std::move(simulation));
    Viewport::get().setCenterInWorldPos(center);
    Viewport::get().setZoomFactor(zoom);
    _originResourceId = errorMessage ? std::nullopt : originResourceId;
    _nameInLibrary = errorMessage ? std::nullopt : nameInLibrary;
    _localFile = errorMessage ? std::nullopt : localFile;
    ++_simulationRevision;
    return errorMessage;
}

std::optional<std::string> const& SimulationGuiService::getOriginResourceId() const
{
    return _originResourceId;
}

void SimulationGuiService::setOriginResourceId(std::optional<std::string> const& value)
{
    _originResourceId = value;
}

std::optional<std::string> const& SimulationGuiService::getNameInLibrary() const
{
    return _nameInLibrary;
}

void SimulationGuiService::setNameInLibrary(std::optional<std::string> const& value)
{
    _nameInLibrary = value;
}

std::optional<std::filesystem::path> const& SimulationGuiService::getLocalFile() const
{
    return _localFile;
}

void SimulationGuiService::setLocalFile(std::optional<std::filesystem::path> const& value)
{
    _localFile = value;
}

int SimulationGuiService::getSimulationRevision() const
{
    return _simulationRevision;
}

void SimulationGuiService::storeAutosaveOrigin() const
{
    auto& settings = GlobalSettings::get();
    auto key = getExitSaveOriginSettingsKey();
    settings.setValue(key + OriginKeySuffix, _originResourceId.value_or(std::string()));
    settings.setValue(key + OriginServerKeySuffix, NetworkService::get().getServerAddress());
    settings.setValue(key + NameInLibraryKeySuffix, _nameInLibrary.value_or(std::string()));
    settings.setValue(key + LocalFileKeySuffix, _localFile.has_value() ? StringHelper::toUtf8(*_localFile) : std::string());
}

void SimulationGuiService::restoreAutosaveOrigin()
{
    auto& settings = GlobalSettings::get();
    auto key = getExitSaveOriginSettingsKey();
    auto originResourceId = settings.getValue(key + OriginKeySuffix, std::string());
    auto originServer = settings.getValue(key + OriginServerKeySuffix, std::string());
    if (!originResourceId.empty() && originServer == NetworkService::get().getServerAddress()) {
        _originResourceId = originResourceId;
    }
    auto nameInLibrary = settings.getValue(key + NameInLibraryKeySuffix, std::string());
    if (!nameInLibrary.empty()) {
        _nameInLibrary = nameInLibrary;
    }
    auto localFile = settings.getValue(key + LocalFileKeySuffix, std::string());
    if (!localFile.empty()) {
        _localFile = StringHelper::toPath(localFile);
    }
}
