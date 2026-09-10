#include "AutosaveController.h"

#include <algorithm>
#include <filesystem>

#include <Base/GlobalSettings.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/SavepointTableService.h>
#include <PersisterInterface/SerializerService.h>
#include <PersisterInterface/TaskProcessor.h>

#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "Viewport.h"

namespace
{
    auto constexpr AutosaveSenderId = "Autosave";
    auto constexpr PeakDetectionInterval = 30;  // In seconds
}

void AutosaveController::init()
{
    _autosaveEnabled = GlobalSettings::get().getValue("windows.autosave.enabled", _autosaveEnabled);
    _autosaveInterval = GlobalSettings::get().getValue("windows.autosave.interval", _autosaveInterval);
    _saveMode = GlobalSettings::get().getValue("windows.autosave.mode", _saveMode);
    _numberOfFiles = GlobalSettings::get().getValue("windows.autosave.number of files", _numberOfFiles);
    _directory = GlobalSettings::get().getValue("windows.autosave.directory", (std::filesystem::current_path() / Const::AutosavePath).string());
    //_catchPeaks = GlobalSettings::get().getValue("windows.autosave.catch peaks", _catchPeaks);

    _lastAutosaveTimepoint = std::chrono::steady_clock::now();
    _lastPeakTimepoint = std::chrono::steady_clock::now();

    _peakProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _peakDeserializedSimulation = std::make_shared<_SharedDeserializedSimulation>();
    updateSavepointTableFromFile();
}

void AutosaveController::shutdown()
{
    GlobalSettings::get().setValue("windows.autosave.enabled", _autosaveEnabled);
    GlobalSettings::get().setValue("windows.autosave.interval", _autosaveInterval);
    GlobalSettings::get().setValue("windows.autosave.mode", _saveMode);
    GlobalSettings::get().setValue("windows.autosave.number of files", _numberOfFiles);
    GlobalSettings::get().setValue("windows.autosave.directory", _directory);
    GlobalSettings::get().setValue("windows.autosave.catch peaks", _catchPeaks);
}

void AutosaveController::process()
{
    processStateUpdates();
    processDeleteNonPersistentSavepoint();
    processCleanup();
    processAutomaticSavepoints();
    _peakProcessor->process();
}

bool AutosaveController::isAutosaveEnabled() const
{
    return _autosaveEnabled;
}

void AutosaveController::setAutosaveEnabled(bool value)
{
    if (value && !_autosaveEnabled) {
        _lastAutosaveTimepoint = std::chrono::steady_clock::now();
    }
    _autosaveEnabled = value;
}

int AutosaveController::getAutosaveInterval() const
{
    return _autosaveInterval;
}

void AutosaveController::setAutosaveInterval(int value)
{
    value = std::max(1, value);
    if (value != _autosaveInterval) {
        _lastAutosaveTimepoint = std::chrono::steady_clock::now();
    }
    _autosaveInterval = value;
}

std::string const& AutosaveController::getDirectory() const
{
    return _directory;
}

void AutosaveController::setDirectory(std::string const& value)
{
    if (value == _directory) {
        return;
    }
    _directory = value;
    updateSavepointTableFromFile();
}

auto AutosaveController::getSaveMode() const -> SaveMode
{
    return _saveMode;
}

void AutosaveController::setSaveMode(SaveMode value)
{
    _saveMode = value;
}

int AutosaveController::getNumberOfFiles() const
{
    return _numberOfFiles;
}

void AutosaveController::setNumberOfFiles(int value)
{
    _numberOfFiles = std::max(1, value);
}

std::optional<SavepointTable> const& AutosaveController::getSavepointTable() const
{
    return _savepointTable;
}

std::chrono::seconds AutosaveController::getDurationUntilNextAutosave() const
{
    auto secondsSinceLastAutosave = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - _lastAutosaveTimepoint);
    return std::chrono::seconds(_autosaveInterval * 60) - secondsSinceLastAutosave;
}

std::optional<SavepointEntry> AutosaveController::getPersistedSavepoint()
{
    auto result = _persistedSavepoint;
    _persistedSavepoint.reset();
    return result;
}

void AutosaveController::onCreateSavepoint(bool usePeakSimulation)
{
    printOverlayMessage("Creating save point ...");

    if (_saveMode == SaveMode_Circular) {
        auto nonPersistentEntries = SavepointTableService::get().truncate(_savepointTable.value(), _numberOfFiles - 1);
        scheduleDeleteNonPersistentSavepoint(nonPersistentEntries);
    }

    PersisterRequestId requestId;
    if (usePeakSimulation && !_peakDeserializedSimulation->isEmpty()) {
        auto senderInfo = SenderInfo{.senderId = SenderId{AutosaveSenderId}, .wishResultData = true, .wishErrorInfo = true};
        auto saveData = SaveDeserializedSimulationRequestData{
            .filename = _directory,
            .sharedDeserializedSimulation = _peakDeserializedSimulation,
            .generateNameFromTimestep = true,
            .resetDeserializedSimulation = true};
        requestId = _PersisterFacade::get()->scheduleSaveDeserializedSimulation(senderInfo, saveData);
    } else {
        auto senderInfo = SenderInfo{.senderId = SenderId{AutosaveSenderId}, .wishResultData = true, .wishErrorInfo = true};
        auto saveData = SaveSimulationRequestData{
            .filename = _directory, .zoom = Viewport::get().getZoomFactor(), .center = Viewport::get().getCenterInWorldPos(), .generateNameFromTimestep = true};
        requestId = _PersisterFacade::get()->scheduleSaveSimulation(senderInfo, saveData);
    }

    auto entry = std::make_shared<_SavepointEntry>(
        _SavepointEntry{.filename = "", .state = SavepointState_InQueue, .timestamp = "", .name = "", .timestep = 0, .requestId = requestId.value});
    SavepointTableService::get().insertEntryAtFront(_savepointTable.value(), entry);
}

void AutosaveController::onDeleteSavepoint(SavepointEntry const& entry)
{
    printOverlayMessage("Deleting save point ...");

    SavepointTableService::get().deleteEntry(_savepointTable.value(), entry);

    if (entry->state != SavepointState_Persisted) {
        scheduleDeleteNonPersistentSavepoint({entry});
    }
}

void AutosaveController::scheduleCleanup()
{
    _scheduleCleanup = true;
}

void AutosaveController::processStateUpdates()
{
    if (_savepointTable.has_value()) {
        for (int row = 0, size = _savepointTable->getSize(); row < size; ++row) {
            updateSavepoint(row);
        }
    }
}

void AutosaveController::processDeleteNonPersistentSavepoint()
{
    std::vector<SavepointEntry> newRequestsToDelete;
    for (auto const& entry : _savepointsInProgressToDelete) {
        if (auto requestState = _PersisterFacade::get()->getRequestState(PersisterRequestId{entry->requestId})) {
            if (requestState.value() == PersisterRequestState::Finished) {
                auto requestResult = _PersisterFacade::get()->fetchPersisterRequestResult(PersisterRequestId{entry->requestId});
                if (auto saveResult = std::dynamic_pointer_cast<_SaveSimulationRequestResult>(requestResult)) {
                    SerializerService::get().deleteSimulation(saveResult->getData().filename);
                }
            } else if (requestState.value() == PersisterRequestState::Error) {
                // Do nothing
            } else {
                newRequestsToDelete.emplace_back(entry);
            }
        }
    }
    _savepointsInProgressToDelete = newRequestsToDelete;
}

void AutosaveController::processCleanup()
{
    if (_scheduleCleanup) {
        printOverlayMessage("Cleaning up save points ...");

        auto nonPersistentEntries = SavepointTableService::get().truncate(_savepointTable.value(), 0);
        scheduleDeleteNonPersistentSavepoint(nonPersistentEntries);
        _scheduleCleanup = false;
    }
}

void AutosaveController::processAutomaticSavepoints()
{
    if (!_autosaveEnabled) {
        return;
    }

    if (!_lastSessionId.has_value() || _lastSessionId.value() != _SimulationFacade::get()->getSessionId()) {
        _lastAutosaveTimepoint = std::chrono::steady_clock::now();
        _lastSessionId = _SimulationFacade::get()->getSessionId();
        _peakDeserializedSimulation->reset();
    }

    auto minSinceLastAutosave = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - _lastAutosaveTimepoint).count();
    if (minSinceLastAutosave >= _autosaveInterval && _savepointTable.has_value()) {
        onCreateSavepoint(_catchPeaks != CatchPeaks_None);
        _lastAutosaveTimepoint = std::chrono::steady_clock::now();
        _lastPeakTimepoint = std::chrono::steady_clock::now();
    }

    if (_catchPeaks != CatchPeaks_None) {
        auto minSinceLastCatchPeak = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - _lastPeakTimepoint).count();
        if (minSinceLastCatchPeak >= PeakDetectionInterval) {
            _peakProcessor->executeTask(
                [&](auto const& senderId) {
                    return _PersisterFacade::get()->scheduleGetPeakSimulation(
                        SenderInfo{.senderId = senderId, .wishResultData = false, .wishErrorInfo = true},
                        GetPeakSimulationRequestData{
                            .peakDeserializedSimulation = _peakDeserializedSimulation,
                            .zoom = Viewport::get().getZoomFactor(),
                            .center = Viewport::get().getCenterInWorldPos()});
                },
                [&](auto const& requestId) {},
                [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
            _lastPeakTimepoint = std::chrono::steady_clock::now();
        }
    }
}

void AutosaveController::scheduleDeleteNonPersistentSavepoint(std::vector<SavepointEntry> const& entries)
{
    for (auto const& entry : entries) {
        if (!entry->requestId.empty() && (entry->state == SavepointState_InQueue || entry->state == SavepointState_InProgress)) {
            _savepointsInProgressToDelete.emplace_back(entry);
        }
    }
}

void AutosaveController::updateSavepoint(int row)
{
    auto state = _savepointTable->at(row)->state;
    if (state != SavepointState_Persisted) {
        auto newEntry = _savepointTable->at(row);
        auto requestState = _PersisterFacade::get()->getRequestState(PersisterRequestId{newEntry->requestId});
        if (requestState.has_value()) {
            if (requestState.value() == PersisterRequestState::InProgress) {
                newEntry->state = SavepointState_InProgress;
            }
            if (requestState.value() == PersisterRequestState::Finished) {
                newEntry->state = SavepointState_Persisted;
                auto requestResult = _PersisterFacade::get()->fetchPersisterRequestResult(PersisterRequestId{newEntry->requestId});

                if (auto saveResult = std::dynamic_pointer_cast<_SaveSimulationRequestResult>(requestResult)) {
                    auto const& data = saveResult->getData();
                    newEntry->timestep = data.timestep;
                    newEntry->timestamp = StringHelper::format(data.timestamp);
                    newEntry->name = data.projectName;
                    newEntry->filename = SavepointTableService::get().calcEntryPath(_savepointTable.value(), data.filename);
                } else if (auto saveResult = std::dynamic_pointer_cast<_SaveDeserializedSimulationRequestResult>(requestResult)) {
                    auto const& data = saveResult->getData();
                    newEntry->timestep = data.timestep;
                    newEntry->timestamp = StringHelper::format(data.timestamp);
                    newEntry->name = data.projectName;
                    newEntry->filename = SavepointTableService::get().calcEntryPath(_savepointTable.value(), data.filename);
                    //newEntry->peak = StringHelper::format(toFloat(sumColorVector(data.statisticsRawData.timeline.timestep.numCellsVariance)), 2);
                    newEntry->peakType = "genome complexity variance";
                }
                _persistedSavepoint = newEntry;
            }
            if (requestState.value() == PersisterRequestState::Error) {
                newEntry->state = SavepointState_Error;
            }
            if (state != newEntry->state) {
                SavepointTableService::get().updateEntry(_savepointTable.value(), row, newEntry);
            }
        }
    }
}

void AutosaveController::updateSavepointTableFromFile()
{
    if (auto savepoint = SavepointTableService::get().loadFromFile(getSavepointFilename()); std::holds_alternative<SavepointTable>(savepoint)) {
        _savepointTable = std::get<SavepointTable>(savepoint);
    } else {
        _savepointTable.reset();
    }
}

std::string AutosaveController::getSavepointFilename() const
{
    return (std::filesystem::path(_directory) / Const::SavepointTableFilename).string();
}
