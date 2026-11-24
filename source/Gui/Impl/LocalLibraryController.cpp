#include "LocalLibraryController.h"

#include <algorithm>

#include <boost/algorithm/string/trim.hpp>

#include <Base/GlobalSettings.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Persister/Interface/LocalLibraryService.h>
#include <Persister/Interface/SerializerService.h>

#include "ConsoleModeController.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "PictureGuiService.h"
#include "SimulationGuiService.h"
#include "Viewport.h"

namespace
{
    auto constexpr SaveSenderId = "LocalLibrarySave";
    auto constexpr LoadSenderId = "LocalLibraryLoad";
    auto constexpr ContentSenderId = "LocalLibraryContent";
    auto constexpr DefaultAutosaveFolderName = "Autosave";
    auto constexpr NumFramesBeforePicture = 3;

    auto constexpr LibraryFolderSettingsKey = "controllers.simulation library.main folder";
    auto constexpr AutosaveIntervalSettingsKey = "controllers.simulation library.autosave interval";
    auto constexpr SaveModeSettingsKey = "controllers.simulation library.save mode";
    auto constexpr NumberOfFilesSettingsKey = "controllers.simulation library.number of files";
    auto constexpr AutosaveFolderSettingsKey = "controllers.simulation library.autosave folder";

    // Settings of the Autosave window of earlier versions
    auto constexpr EarlierAutosaveIntervalSettingsKey = "windows.autosave.interval";
    auto constexpr EarlierSaveModeSettingsKey = "windows.autosave.mode";
    auto constexpr EarlierNumberOfFilesSettingsKey = "windows.autosave.number of files";
    auto constexpr EarlierAutosaveDirectorySettingsKey = "windows.autosave.directory";

    auto constexpr InvalidNameMessage = "The name is not valid. The name and each folder in front of it must not be empty or longer than 100 characters, "
                                        "must not begin or end with a dot or a space and must not contain any of the characters "
                                        "< > : \" \\ | ? *. The name of a simulation must also not end with .genome.";
    auto constexpr InvalidGenomeNameMessage = "The name is not valid.";
    auto constexpr TakenNameMessage = "A file, a picture or a description with this name already exists.";
    auto constexpr RenamingFailedMessage = "The renaming failed. The new name may not be valid, or one of the files may be in use.";
    auto constexpr BusyMessage = "A simulation cannot be changed while it is being saved or loaded.";
    auto constexpr MissingFileMessage = "The file no longer exists.";

    std::vector<std::filesystem::path> getEarlierAutosaveFolders()
    {
        std::vector<std::filesystem::path> result;
        auto earlierDirectory = GlobalSettings::get().getValue(EarlierAutosaveDirectorySettingsKey, std::string());
        if (!earlierDirectory.empty()) {
            result.emplace_back(StringHelper::toPath(earlierDirectory));
        }
        result.emplace_back(std::filesystem::current_path() / Const::AutosavePath);
        return result;
    }

    bool containsSavePoints(std::filesystem::path const& folder)
    {
        auto content = LocalLibraryService::get().loadContent(folder);
        return std::ranges::any_of(content.simulations, [](LibraryFile const& simulation) {
            return simulation.kind == LibraryFileKind::Autosave && simulation.name.find('/') == std::string::npos;
        });
    }

    // Some versions saved into the resource folder of the installation, whose folders must not become folders of the library. The folder
    // may belong to another installation, which is why its name counts as well.
    bool isResourceFolder(std::filesystem::path const& folder)
    {
        auto& service = LocalLibraryService::get();
        return service.isInside(std::filesystem::current_path() / Const::ResourcePath, folder) || service.normalize(folder).filename() == Const::ResourcePath;
    }

    // The folder in which the Autosave window of earlier versions has created save points becomes the library, which keeps them at hand
    std::filesystem::path getInitialLibraryFolder()
    {
        for (auto const& earlierFolder : getEarlierAutosaveFolders()) {
            if (!isResourceFolder(earlierFolder) && containsSavePoints(earlierFolder)) {
                return earlierFolder;
            }
        }
        return std::filesystem::current_path() / Const::LocalLibraryPath;
    }

    // The autosaves of earlier versions are continued in their folder if it belongs to the library
    std::string getInitialAutosaveFolderName(std::filesystem::path const& libraryFolder)
    {
        auto& service = LocalLibraryService::get();
        for (auto const& earlierFolder : getEarlierAutosaveFolders()) {
            auto folderName = service.getFolderName(libraryFolder, earlierFolder);
            if (folderName.has_value() && (folderName->empty() || service.isValidFolderName(*folderName))) {
                return *folderName;
            }
        }
        return DefaultAutosaveFolderName;
    }

    // Nested folders are separated by a slash or a backslash, and a separator at the end is still being typed
    std::optional<std::string> normalizeAutosaveFolderName(std::string const& value)
    {
        auto folderName = boost::algorithm::trim_copy(value);
        std::ranges::replace(folderName, '\\', '/');
        boost::algorithm::trim_right_if(folderName, [](char character) { return character == '/' || character == ' '; });
        if (!folderName.empty() && !LocalLibraryService::get().isValidFolderName(folderName)) {
            return std::nullopt;
        }
        return folderName;
    }
}

void LocalLibraryController::init()
{
    auto& settings = GlobalSettings::get();
    auto& service = LocalLibraryService::get();

    auto libraryFolder = settings.getValue(LibraryFolderSettingsKey, std::string());
    _libraryFolder = service.normalize(libraryFolder.empty() ? getInitialLibraryFolder() : StringHelper::toPath(libraryFolder));
    _autosaveInterval = std::max(1, settings.getValue(AutosaveIntervalSettingsKey, settings.getValue(EarlierAutosaveIntervalSettingsKey, _autosaveInterval)));
    auto saveMode = settings.getValue(SaveModeSettingsKey, settings.getValue(EarlierSaveModeSettingsKey, _saveMode));
    _saveMode = saveMode == SaveMode_Unlimited ? SaveMode_Unlimited : SaveMode_Limited;
    _numberOfFiles = std::max(1, settings.getValue(NumberOfFilesSettingsKey, settings.getValue(EarlierNumberOfFilesSettingsKey, _numberOfFiles)));
    if (!setAutosaveFolderName(settings.getValue(AutosaveFolderSettingsKey, getInitialAutosaveFolderName(_libraryFolder)))) {
        _autosaveFolderName = DefaultAutosaveFolderName;
    }

    _lastAutosaveTimepoint = std::chrono::steady_clock::now();
}

void LocalLibraryController::shutdown()
{
    auto& settings = GlobalSettings::get();
    settings.setValue(LibraryFolderSettingsKey, StringHelper::toUtf8(_libraryFolder));
    settings.setValue(AutosaveIntervalSettingsKey, _autosaveInterval);
    settings.setValue(SaveModeSettingsKey, _saveMode);
    settings.setValue(NumberOfFilesSettingsKey, _numberOfFiles);
    settings.setValue(AutosaveFolderSettingsKey, _autosaveFolderName);
}

void LocalLibraryController::process()
{
    processPendingSaves();
    processPendingLoads();
    processPendingPicture();
    processAutosave();
    processContent();
}

std::filesystem::path const& LocalLibraryController::getLibraryFolder() const
{
    return _libraryFolder;
}

void LocalLibraryController::setLibraryFolder(std::filesystem::path const& value)
{
    if (value.empty()) {
        return;
    }
    auto libraryFolder = LocalLibraryService::get().normalize(value);
    if (libraryFolder == _libraryFolder) {
        return;
    }
    _libraryFolder = libraryFolder;

    // The name of the running simulation belongs to the previous library
    SimulationGuiService::get().setNameInLibrary(std::nullopt);
    _contentOutdated = true;
}

LocalLibraryContent const& LocalLibraryController::getContent() const
{
    return _content;
}

int LocalLibraryController::getRevision() const
{
    return _revision;
}

void LocalLibraryController::refresh()
{
    _contentOutdated = true;
}

bool LocalLibraryController::isAutosaveEnabled() const
{
    return _autosaveEnabled;
}

void LocalLibraryController::setAutosaveEnabled(bool value)
{
    if (value && !_autosaveEnabled) {
        _lastAutosaveTimepoint = std::chrono::steady_clock::now();
    }
    _autosaveEnabled = value;
}

int LocalLibraryController::getAutosaveInterval() const
{
    return _autosaveInterval;
}

void LocalLibraryController::setAutosaveInterval(int value)
{
    value = std::max(1, value);
    if (value != _autosaveInterval) {
        _lastAutosaveTimepoint = std::chrono::steady_clock::now();
    }
    _autosaveInterval = value;
}

auto LocalLibraryController::getSaveMode() const -> SaveMode
{
    return _saveMode;
}

void LocalLibraryController::setSaveMode(SaveMode value)
{
    _saveMode = value;
}

int LocalLibraryController::getNumberOfFiles() const
{
    return _numberOfFiles;
}

void LocalLibraryController::setNumberOfFiles(int value)
{
    _numberOfFiles = std::max(1, value);
}

std::string const& LocalLibraryController::getAutosaveFolderName() const
{
    return _autosaveFolderName;
}

bool LocalLibraryController::setAutosaveFolderName(std::string const& value)
{
    auto folderName = normalizeAutosaveFolderName(value);
    if (!folderName.has_value()) {
        return false;
    }
    _autosaveFolderName = *folderName;
    return true;
}

bool LocalLibraryController::isAutosaveFolder(std::filesystem::path const& folder) const
{
    std::error_code error;
    return std::filesystem::equivalent(folder, getAutosaveFolder(), error) && !error;
}

std::optional<std::filesystem::path> LocalLibraryController::resolveAutosaveFolder(std::string const& folderName) const
{
    auto normalizedFolderName = normalizeAutosaveFolderName(folderName);
    if (!normalizedFolderName.has_value()) {
        return std::nullopt;
    }
    return normalizedFolderName->empty() ? _libraryFolder : _libraryFolder / StringHelper::toPath(*normalizedFolderName);
}

std::chrono::seconds LocalLibraryController::getDurationUntilNextAutosave() const
{
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - _lastAutosaveTimepoint);
    return std::max(std::chrono::seconds(0), std::chrono::seconds(std::chrono::minutes(_autosaveInterval)) - elapsed);
}

bool LocalLibraryController::isSaving() const
{
    return !_pendingSaves.empty();
}

void LocalLibraryController::onSave(std::string const& name, std::optional<std::string> const& description, std::string const& jpg)
{
    auto& service = LocalLibraryService::get();
    if (!service.isValidSimulationName(name)) {
        GenericMessageDialog::get().information("Error", InvalidNameMessage);
        return;
    }
    scheduleSave(PendingSave{
        .target = service.getSimulationPath(_libraryFolder, name),
        .name = name,
        .description = description,
        .jpg = jpg.empty() ? std::nullopt : std::make_optional(jpg)});
}

void LocalLibraryController::onReplace(std::filesystem::path const& simulationPath, std::string const& jpg)
{
    auto& service = LocalLibraryService::get();
    auto name = service.getSimulationName(_libraryFolder, simulationPath);
    scheduleSave(PendingSave{
        .target = simulationPath,
        .name = name.has_value() && service.isValidSimulationName(*name) ? name : std::nullopt,
        .jpg = jpg.empty() ? std::nullopt : std::make_optional(jpg)});
}

void LocalLibraryController::onSaveToFile(std::filesystem::path const& filename)
{
    // A file inside the library is a simulation of the library and gets its picture
    auto& service = LocalLibraryService::get();
    auto name = service.getSimulationName(_libraryFolder, filename);
    auto belongsToLibrary = name.has_value() && service.isValidSimulationName(*name) && service.getSimulationPath(_libraryFolder, *name) == filename;
    scheduleSave(PendingSave{.target = filename, .name = belongsToLibrary ? name : std::nullopt});
}

void LocalLibraryController::onAutosave()
{
    scheduleSave(PendingSave{.target = getAutosaveFolder(), .automatic = true});
    _lastAutosaveTimepoint = std::chrono::steady_clock::now();
}

void LocalLibraryController::onLoad(std::filesystem::path const& simulationPath)
{
    printOverlayMessage("Loading ...");

    auto senderInfo = SenderInfo{.senderId = SenderId{LoadSenderId}, .wishResultData = true, .wishErrorInfo = true};
    auto readData = ReadSimulationRequestData{.filename = simulationPath};
    _pendingLoads.emplace_back(
        PendingLoad{.requestId = _PersisterFacade::get()->scheduleReadSimulation(senderInfo, readData), .simulationPath = simulationPath});
}

std::optional<std::string>
LocalLibraryController::onSaveGenome(std::string const& name, GenomeDesc const& genome, std::optional<std::string> const& description, std::string const& jpg)
{
    auto& service = LocalLibraryService::get();
    if (!service.isValidGenomeName(name)) {
        return std::string(InvalidGenomeNameMessage);
    }
    auto genomePath = service.getGenomePath(_libraryFolder, name);
    std::error_code error;
    std::filesystem::create_directories(genomePath.parent_path(), error);
    if (!SerializerService::get().serializeGenomeToFile(genomePath, genome)) {
        return "The genome could not be saved in " + StringHelper::toUtf8(genomePath) + ".";
    }
    // A replaced genome must not keep the picture of its predecessor
    if (jpg.empty()) {
        std::filesystem::remove(service.getPicturePath(genomePath), error);
    } else {
        service.savePicture(genomePath, jpg);
    }
    if (description.has_value()) {
        service.saveDescription(genomePath, *description);
    }
    _contentOutdated = true;
    return std::nullopt;
}

std::optional<std::string>
LocalLibraryController::onEdit(std::filesystem::path const& simulationPath, std::string const& newName, std::optional<std::string> const& newDescription)
{
    auto& service = LocalLibraryService::get();
    if (isTransferring(simulationPath)) {
        return std::string(BusyMessage);
    }

    // An autosave may have been removed while it was edited
    std::error_code error;
    if (!std::filesystem::exists(simulationPath, error)) {
        _contentOutdated = true;
        return std::string(MissingFileMessage);
    }
    auto newSimulationPath = simulationPath;
    if (service.getSimulationName(_libraryFolder, simulationPath) != newName) {
        if (auto errorMessage = rename(simulationPath, newName, newSimulationPath)) {
            return errorMessage;
        }
    }
    _contentOutdated = true;
    if (newDescription.has_value() && !service.saveDescription(newSimulationPath, *newDescription)) {
        return "The description could not be saved in " + StringHelper::toUtf8(service.getDescriptionPath(newSimulationPath)) + ".";
    }
    return std::nullopt;
}

std::optional<std::string> LocalLibraryController::onRename(std::vector<Renaming> const& renamings)
{
    if (std::ranges::any_of(renamings, [this](Renaming const& renaming) { return isTransferring(renaming.simulationPath); })) {
        return std::string(BusyMessage);
    }
    std::optional<std::string> result;
    for (auto const& renaming : renamings) {
        std::filesystem::path newSimulationPath;
        if (auto errorMessage = rename(renaming.simulationPath, renaming.newName, newSimulationPath)) {
            result = errorMessage;
        }
    }
    _contentOutdated = true;
    return result;
}

std::optional<std::string> LocalLibraryController::onDelete(std::vector<std::filesystem::path> const& simulationPaths)
{
    if (std::ranges::any_of(simulationPaths, [this](std::filesystem::path const& simulationPath) { return isTransferring(simulationPath); })) {
        return std::string(BusyMessage);
    }
    auto& service = LocalLibraryService::get();
    auto const& nameOfRunningSimulation = SimulationGuiService::get().getNameInLibrary();
    for (auto const& simulationPath : simulationPaths) {
        if (service.isGenomePath(simulationPath)) {
            service.deleteGenome(_libraryFolder, simulationPath);
            continue;
        }
        if (nameOfRunningSimulation.has_value() && service.getSimulationName(_libraryFolder, simulationPath) == nameOfRunningSimulation) {
            SimulationGuiService::get().setNameInLibrary(std::nullopt);
        }
        if (auto const& localFile = SimulationGuiService::get().getLocalFile(); localFile.has_value() && *localFile == service.normalize(simulationPath)) {
            SimulationGuiService::get().setLocalFile(std::nullopt);
        }
        service.deleteSimulation(_libraryFolder, simulationPath);
    }
    _contentOutdated = true;
    return std::nullopt;
}

std::optional<std::string> LocalLibraryController::fetchSavedFilename()
{
    auto result = _savedFilename;
    _savedFilename.reset();
    return result;
}

void LocalLibraryController::processPendingSaves()
{
    std::vector<PendingSave> remainingSaves;
    for (auto const& pendingSave : _pendingSaves) {
        auto requestState = _PersisterFacade::get()->getRequestState(pendingSave.requestId);
        if (!requestState.has_value()) {
            continue;
        }
        if (requestState.value() == PersisterRequestState::Finished) {
            onSaveFinished(pendingSave, _PersisterFacade::get()->fetchSaveSimulationData(pendingSave.requestId));
        } else if (requestState.value() == PersisterRequestState::Error) {
            GenericMessageDialog::get().information("Error", _PersisterFacade::get()->fetchError(pendingSave.requestId).message);
        } else {
            remainingSaves.emplace_back(pendingSave);
        }
    }
    _pendingSaves = remainingSaves;
}

void LocalLibraryController::processPendingLoads()
{
    // Loading replaces the running simulation, which must not happen before it is saved
    if (!_pendingSaves.empty()) {
        return;
    }
    auto& service = LocalLibraryService::get();

    std::vector<PendingLoad> remainingLoads;
    for (auto const& pendingLoad : _pendingLoads) {
        auto requestState = _PersisterFacade::get()->getRequestState(pendingLoad.requestId);
        if (!requestState.has_value()) {
            continue;
        }
        if (requestState.value() == PersisterRequestState::Error) {
            GenericMessageDialog::get().information("Error", _PersisterFacade::get()->fetchError(pendingLoad.requestId).message);
            continue;
        }
        if (requestState.value() != PersisterRequestState::Finished) {
            remainingLoads.emplace_back(pendingLoad);
            continue;
        }

        // An opened autosave is saved under a name of its own by default, since autosave deletes the files with its name. A file with a name
        // that could not be given has no name under which it could be saved again.
        auto name = service.getSimulationName(_libraryFolder, pendingLoad.simulationPath);
        auto isAutosave = service.isAutosaveName(StringHelper::toUtf8(pendingLoad.simulationPath.stem()));
        auto nameInLibrary = name.has_value() && service.isValidSimulationName(*name) && !isAutosave ? name : std::nullopt;
        auto data = _PersisterFacade::get()->fetchReadSimulationData(pendingLoad.requestId);
        if (auto errorMessage = SimulationGuiService::get().loadSimulation(
                std::move(data.simulationDesc), std::nullopt, nameInLibrary, service.normalize(pendingLoad.simulationPath))) {
            GenericMessageDialog::get().information("Error", *errorMessage);
            continue;
        }
        printOverlayMessage(StringHelper::toUtf8(pendingLoad.simulationPath.filename()));

        // A simulation without a picture, like those saved by earlier versions, gets one when it is loaded
        std::error_code error;
        if (name.has_value() && !ConsoleModeController::get().isActive()
            && !std::filesystem::exists(service.getPicturePath(pendingLoad.simulationPath), error)) {
            _pendingPicture = PendingPicture{
                .simulationPath = pendingLoad.simulationPath, .sessionId = _SimulationFacade::get()->getSessionId(), .numFramesToWait = NumFramesBeforePicture};
        }
    }
    _pendingLoads = remainingLoads;
}

void LocalLibraryController::processPendingPicture()
{
    if (!_pendingPicture.has_value()) {
        return;
    }
    if (_pendingPicture->sessionId != _SimulationFacade::get()->getSessionId() || ConsoleModeController::get().isActive()) {
        _pendingPicture.reset();
        return;
    }

    // The picture is taken after the loaded simulation has been rendered
    if (--_pendingPicture->numFramesToWait > 0) {
        return;
    }
    std::error_code error;
    if (std::filesystem::exists(_pendingPicture->simulationPath, error)) {
        if (auto jpg = PictureGuiService::get().createSimulationPreviewJpg()) {
            LocalLibraryService::get().savePicture(_pendingPicture->simulationPath, *jpg);
            _contentOutdated = true;
        }
    }
    _pendingPicture.reset();
}

void LocalLibraryController::processAutosave()
{
    // Autosave belongs to one simulation, so a created or loaded one starts without it
    auto simulationRevision = SimulationGuiService::get().getSimulationRevision();
    if (simulationRevision != _lastSimulationRevision) {
        _lastSimulationRevision = simulationRevision;
        _autosaveEnabled = false;
    }
    if (!_autosaveEnabled) {
        return;
    }
    auto sessionId = _SimulationFacade::get()->getSessionId();
    if (!_lastSessionId.has_value() || _lastSessionId.value() != sessionId) {
        _lastAutosaveTimepoint = std::chrono::steady_clock::now();
        _lastSessionId = sessionId;
    }
    if (std::chrono::steady_clock::now() - _lastAutosaveTimepoint >= std::chrono::minutes(_autosaveInterval)) {
        onAutosave();
    }
}

void LocalLibraryController::processContent()
{
    if (_contentRequestId.has_value()) {
        auto requestState = _PersisterFacade::get()->getRequestState(*_contentRequestId);
        if (requestState == PersisterRequestState::InQueue || requestState == PersisterRequestState::InProgress) {
            return;
        }
        if (requestState == PersisterRequestState::Error) {
            GenericMessageDialog::get().information("Error", _PersisterFacade::get()->fetchError(*_contentRequestId).message);
        } else if (requestState == PersisterRequestState::Finished) {
            auto data = _PersisterFacade::get()->fetchReadLocalLibraryData(*_contentRequestId);

            // The result is dropped if the files or the library folder have changed during the reading, which is then repeated
            if (!_contentOutdated) {
                _content = std::move(data.content);

                // A folder may have been renamed to another spelling of its name, which the path of the running simulation follows
                if (auto const& localFile = SimulationGuiService::get().getLocalFile()) {
                    SimulationGuiService::get().setLocalFile(LocalLibraryService::get().normalize(*localFile));
                }
                ++_revision;
            }
        }
        _contentRequestId.reset();
    }

    if (_contentOutdated) {
        auto senderInfo = SenderInfo{.senderId = SenderId{ContentSenderId}, .wishResultData = true, .wishErrorInfo = true};
        auto requestData = ReadLocalLibraryRequestData{.libraryFolder = _libraryFolder};
        _contentRequestId = _PersisterFacade::get()->scheduleReadLocalLibrary(senderInfo, requestData);
        _contentOutdated = false;
    }
}

void LocalLibraryController::scheduleSave(PendingSave pendingSave)
{
    auto folder = pendingSave.automatic ? pendingSave.target : pendingSave.target.parent_path();
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (!std::filesystem::is_directory(folder, error)) {
        GenericMessageDialog::get().information("Error", "The folder " + StringHelper::toUtf8(folder) + " could not be created.");
        return;
    }

    // Only the simulations of the library get a picture. The rendering is switched off in the console mode.
    auto belongsToLibrary = pendingSave.automatic || pendingSave.name.has_value();
    if (belongsToLibrary && !pendingSave.jpg.has_value() && !ConsoleModeController::get().isActive()) {
        pendingSave.jpg = PictureGuiService::get().createSimulationPreviewJpg();
    }
    pendingSave.sessionId = _SimulationFacade::get()->getSessionId();

    auto senderInfo = SenderInfo{.senderId = SenderId{SaveSenderId}, .wishResultData = true, .wishErrorInfo = true};
    auto saveData = SaveSimulationRequestData{
        .filename = pendingSave.target,
        .zoom = Viewport::get().getZoomFactor(),
        .center = Viewport::get().getCenterInWorldPos(),
        .generateNameFromTimestep = pendingSave.automatic,
        .generatedNamePrefix = Const::AutosavePrefix};
    pendingSave.requestId = _PersisterFacade::get()->scheduleSaveSimulation(senderInfo, saveData);
    printOverlayMessage(pendingSave.automatic ? "Autosaving ..." : "Saving ...");
    _pendingSaves.emplace_back(std::move(pendingSave));
}

void LocalLibraryController::onSaveFinished(PendingSave const& pendingSave, SaveSimulationResultData const& data)
{
    auto& service = LocalLibraryService::get();
    if (pendingSave.jpg.has_value()) {
        service.savePicture(data.filename, *pendingSave.jpg);
    }
    if (pendingSave.name.has_value()) {
        if (pendingSave.description.has_value()) {
            service.saveDescription(data.filename, *pendingSave.description);
        }

        // Another simulation may have been loaded or created in the meantime
        if (pendingSave.sessionId == _SimulationFacade::get()->getSessionId()) {
            SimulationGuiService::get().setNameInLibrary(pendingSave.name);
            SimulationGuiService::get().setLocalFile(service.normalize(data.filename));
        }
    }
    if (pendingSave.automatic && _saveMode == SaveMode_Limited) {
        service.removeExcessAutosaves(pendingSave.target, _numberOfFiles);
    }
    _savedFilename = StringHelper::toUtf8(data.filename);
    _contentOutdated = true;
}

std::filesystem::path LocalLibraryController::getAutosaveFolder() const
{
    return _autosaveFolderName.empty() ? _libraryFolder : (_libraryFolder / StringHelper::toPath(_autosaveFolderName)).make_preferred();
}

std::optional<std::string>
LocalLibraryController::rename(std::filesystem::path const& simulationPath, std::string const& newName, std::filesystem::path& newSimulationPath)
{
    auto& service = LocalLibraryService::get();
    auto isGenome = service.isGenomePath(simulationPath);
    auto name = service.getSimulationName(_libraryFolder, simulationPath);
    auto const& localFile = SimulationGuiService::get().getLocalFile();
    auto isLocalFile = localFile.has_value() && *localFile == service.normalize(simulationPath);
    auto isNameTaken =
        isGenome ? service.isGenomeNameTaken(_libraryFolder, newName, simulationPath) : service.isSimulationNameTaken(_libraryFolder, newName, simulationPath);
    if (isNameTaken) {
        return std::string(TakenNameMessage);
    }
    auto movedSimulationPath =
        isGenome ? service.moveGenome(_libraryFolder, simulationPath, newName) : service.moveSimulation(_libraryFolder, simulationPath, newName);
    if (!movedSimulationPath.has_value()) {
        return std::string(RenamingFailedMessage);
    }
    newSimulationPath = *movedSimulationPath;
    if (isGenome) {
        return std::nullopt;
    }

    if (name.has_value() && name == SimulationGuiService::get().getNameInLibrary()) {
        SimulationGuiService::get().setNameInLibrary(service.isValidSimulationName(newName) ? std::make_optional(newName) : std::nullopt);
    }
    if (isLocalFile) {
        SimulationGuiService::get().setLocalFile(service.normalize(newSimulationPath));
    }
    if (_pendingPicture.has_value() && _pendingPicture->simulationPath == simulationPath) {
        _pendingPicture->simulationPath = newSimulationPath;
    }
    return std::nullopt;
}

bool LocalLibraryController::isTransferring(std::filesystem::path const& simulationPath) const
{
    auto& service = LocalLibraryService::get();
    auto normalizedPath = service.normalize(simulationPath);
    return std::ranges::any_of(_pendingSaves, [&](PendingSave const& pendingSave) { return service.normalize(pendingSave.target) == normalizedPath; })
        || std::ranges::any_of(_pendingLoads, [&](PendingLoad const& pendingLoad) { return service.normalize(pendingLoad.simulationPath) == normalizedPath; });
}
