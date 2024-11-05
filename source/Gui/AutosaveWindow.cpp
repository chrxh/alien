#include "AutosaveWindow.h"

#include <filesystem>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/Resources.h"
#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "PersisterInterface/SavepointTableService.h"
#include "PersisterInterface/SerializerService.h"
#include "PersisterInterface/TaskProcessor.h"

#include "AlienImGui.h"
#include "FileTransferController.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto constexpr RightColumnWidth = 200.0f;
    auto constexpr AutosaveSenderId = "Autosave";
    auto constexpr PeakDetectionInterval = 30;  //in seconds
}

AutosaveWindow::AutosaveWindow()
    : AlienWindow("Autosave", "windows.autosave", false)
{}

void AutosaveWindow::initIntern(SimulationFacade simulationFacade, PersisterFacade persisterFacade)
{
    _simulationFacade = simulationFacade;
    _persisterFacade = persisterFacade;
    _settingsOpen = GlobalSettings::get().getValue("windows.autosave.settings.open", _settingsOpen);
    _settingsHeight = GlobalSettings::get().getValue("windows.autosave.settings.height", _settingsHeight);
    _autosaveEnabled = GlobalSettings::get().getValue("windows.autosave.enabled", _autosaveEnabled);
    _origAutosaveInterval = GlobalSettings::get().getValue("windows.autosave.interval", _origAutosaveInterval);
    _autosaveInterval = _origAutosaveInterval;

    _origSaveMode = GlobalSettings::get().getValue("windows.autosave.mode", _origSaveMode);
    _saveMode = _origSaveMode;

    _origNumberOfFiles = GlobalSettings::get().getValue("windows.autosave.number of files", _origNumberOfFiles);
    _numberOfFiles = _origNumberOfFiles;

    _origDirectory = GlobalSettings::get().getValue("windows.autosave.directory", (std::filesystem::current_path() / Const::BasePath).string());
    _directory = _origDirectory;

    _origCatchPeaks = GlobalSettings::get().getValue("windows.autosave.catch peaks", _origCatchPeaks);
    _catchPeaks = _origCatchPeaks;

    _lastAutosaveTimepoint = std::chrono::steady_clock::now();
    _lastPeakTimepoint = std::chrono::steady_clock::now();

    _peakProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);
    _peakDeserializedSimulation = std::make_shared<_SharedDeserializedSimulation>();
    updateSavepointTableFromFile();
}

void AutosaveWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.autosave.settings.open", _settingsOpen);
    GlobalSettings::get().setValue("windows.autosave.settings.height", _settingsHeight);
    GlobalSettings::get().setValue("windows.autosave.enabled", _autosaveEnabled);
    GlobalSettings::get().setValue("windows.autosave.interval", _autosaveInterval);
    GlobalSettings::get().setValue("windows.autosave.mode", _saveMode);
    GlobalSettings::get().setValue("windows.autosave.number of files", _numberOfFiles);
    GlobalSettings::get().setValue("windows.autosave.directory", _directory);
    GlobalSettings::get().setValue("windows.autosave.catch peaks", _catchPeaks);
}

void AutosaveWindow::processIntern()
{
    try {
        processToolbar();

        if (ImGui::BeginChild("##child1", {0, -scale(44.0f)})) {
            processHeader();

            //AlienImGui::Separator();
            if (ImGui::BeginChild("##child2", {0, _settingsOpen ? -scale(_settingsHeight) : -scale(50.0f)})) {
                processTable();
            }
            ImGui::EndChild();

            processSettings();
        }
        ImGui::EndChild();

        processStatusBar();

        validationAndCorrection();
    } catch (std::runtime_error const& error) {
        GenericMessageDialog::get().information("Error", error.what());
    }
}

void AutosaveWindow::processBackground()
{
    processDeleteNonPersistentSavepoint();
    processCleanup();
    processAutomaticSavepoints();
    _peakProcessor->process();
}

void AutosaveWindow::processToolbar()
{
    ImGui::SameLine();
    ImGui::BeginDisabled(!_savepointTable.has_value());
    if (AlienImGui::ToolbarButton(ICON_FA_PLUS)) {
        onCreateSavepoint(false);
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Create save point");

    ImGui::SameLine();
    ImGui::BeginDisabled(!static_cast<bool>(_selectedEntry) || (_selectedEntry->state != SavepointState_Persisted && !_selectedEntry->requestId.empty()));
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS)) {
        onDeleteSavepoint(_selectedEntry);
    }
    AlienImGui::Tooltip("Delete save point");
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!_savepointTable.has_value() || _savepointTable->isEmpty());
    if (AlienImGui::ToolbarButton(ICON_FA_BROOM)) {
        GenericMessageDialog::get().yesNo("Delete", "Do you really want to delete the all savepoints?", [&]() { scheduleCleanup(); });
    }
    AlienImGui::Tooltip("Delete all save points");
    ImGui::EndDisabled();

    AlienImGui::Separator();
}

void AutosaveWindow::processHeader()
{
}

void AutosaveWindow::processTable()
{
    if (!_savepointTable.has_value()) {
        AlienImGui::Text("Error: Savepoint files could not be read or created in the specified directory.");
        return;
    }
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Save files", 4, flags, ImVec2(0, 0), 0.0f)) {
        ImGui::TableSetupColumn("Simulation", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Timestamp", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Time step", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
        ImGui::TableSetupColumn("Peak value", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(200.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(_savepointTable->getSize());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                updateSavepoint(row);
                auto const& entry = _savepointTable->at(row);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(ImGui::GetTextLineHeightWithSpacing()));

                // project name
                ImGui::TableNextColumn();
                if (entry->state == SavepointState_InQueue) {
                    AlienImGui::Text("In queue");
                }
                if (entry->state == SavepointState_InProgress) {
                    AlienImGui::Text("In progress");
                }
                if (entry->state == SavepointState_Persisted) {
                    auto triggerLoadSavepoint = AlienImGui::ActionButton(AlienImGui::ActionButtonParameters().buttonText(ICON_FA_DOWNLOAD));
                    AlienImGui::Tooltip("Load save point", false);
                    if (triggerLoadSavepoint) {
                        onLoadSavepoint(entry);
                    }

                    ImGui::SameLine();
                    AlienImGui::Text(entry->name);
                }
                if (entry->state == SavepointState_Error) {
                    AlienImGui::Text("Error");
                }

                ImGui::SameLine();
                auto selected = _selectedEntry == entry;
                if (ImGui::Selectable(
                        "",
                        &selected,
                        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                        ImVec2(0, scale(ImGui::GetTextLineHeightWithSpacing()) - ImGui::GetStyle().FramePadding.y))) {
                    _selectedEntry = selected ? entry : nullptr;
                }

                // timestamp
                ImGui::TableNextColumn();
                if (entry->state == SavepointState_Persisted) {
                    AlienImGui::Text(entry->timestamp);
                }

                // timestep
                ImGui::TableNextColumn();
                if (entry->state == SavepointState_Persisted) {
                    AlienImGui::Text(StringHelper::format(entry->timestep));
                }

                // peak
                ImGui::TableNextColumn();
                AlienImGui::Text(entry->peak);

                if (!entry->peakType.empty()) {
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, Const::BrowserResourcePropertiesTextColor.Value);
                    AlienImGui::Text(" (" + entry->peakType + ")");
                    ImGui::PopStyleColor();
                }

                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
}

void AutosaveWindow::processSettings()
{
    if (_settingsOpen) {
        ImGui::Spacing();
        ImGui::Spacing();
        AlienImGui::MovableSeparator(_settingsHeight);
    } else {
        AlienImGui::Separator();
    }

    _settingsOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Settings").highlighted(true).defaultOpen(_settingsOpen));
    if (_settingsOpen) {
        if (ImGui::BeginChild("##addons", {scale(0), 0})) {
            if (AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Autosave interval (min)").textWidth(RightColumnWidth).defaultValue(_origAutosaveInterval),
                    _autosaveInterval,
                    &_autosaveEnabled)) {
                if (_autosaveEnabled) {
                    _lastAutosaveTimepoint = std::chrono::steady_clock::now();
                }
            }
            if (AlienImGui::Switcher(
                    AlienImGui::SwitcherParameters()
                    .name("Catch peaks")
                    .textWidth(RightColumnWidth)
                    .defaultValue(_origCatchPeaks)
                    .disabled(!_autosaveEnabled)
                    .values({
                        "None",
                        "Genome complexity variance",
                    }),
                _catchPeaks)) {
                _peakDeserializedSimulation->setDeserializedSimulation(DeserializedSimulation());
            }

            if (AlienImGui::InputText(
                    AlienImGui::InputTextParameters().name("Directory").textWidth(RightColumnWidth).defaultValue(_origDirectory).folderButton(true),
                    _directory)) {
                updateSavepointTableFromFile();
            }
            AlienImGui::Switcher(
                AlienImGui::SwitcherParameters()
                    .name("Mode")
                    .values({"Limited save files", "Unlimited save files"})
                    .textWidth(RightColumnWidth)
                    .defaultValue(_origSaveMode),
                _saveMode);
            if (_saveMode == SaveMode_Circular) {
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Number of files").textWidth(RightColumnWidth).defaultValue(_origNumberOfFiles), _numberOfFiles);
            }
        }
        ImGui::EndChild();
        AlienImGui::EndTreeNode();
    }
}

void AutosaveWindow::processStatusBar()
{
    std::string statusText = " " ICON_FA_INFO_CIRCLE " ";
    statusText += [&] {
        if (!_autosaveEnabled) {
            return std::string("No autosave scheduled");
        }
        auto secondsSinceLastAutosave = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - _lastAutosaveTimepoint);
        return "Next autosave in " + StringHelper::format(std::chrono::seconds(_autosaveInterval * 60) - secondsSinceLastAutosave);
    }();
    AlienImGui::StatusBar(statusText);
}

void AutosaveWindow::onCreateSavepoint(bool usePeakSimulation)
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
            .filename = _directory, .sharedDeserializedSimulation = _peakDeserializedSimulation, .generateNameFromTimestep = true, .resetDeserializedSimulation = true};
        requestId = _persisterFacade->scheduleSaveDeserializedSimulation(senderInfo, saveData);
    } else {
        auto senderInfo = SenderInfo{.senderId = SenderId{AutosaveSenderId}, .wishResultData = true, .wishErrorInfo = true};
        auto saveData = SaveSimulationRequestData{
            .filename = _directory, .zoom = Viewport::get().getZoomFactor(), .center = Viewport::get().getCenterInWorldPos(), .generateNameFromTimestep = true};
        requestId = _persisterFacade->scheduleSaveSimulation(senderInfo, saveData);
    }

    auto entry = std::make_shared<_SavepointEntry>(
        _SavepointEntry{.filename = "", .state = SavepointState_InQueue, .timestamp = "", .name = "", .timestep = 0, .requestId = requestId.value});
    SavepointTableService::get().insertEntryAtFront(_savepointTable.value(), entry);
}

void AutosaveWindow::onDeleteSavepoint(SavepointEntry const& entry)
{
    printOverlayMessage("Deleting save point ...");

    SavepointTableService::get().deleteEntry(_savepointTable.value(), entry);

    if (entry->state != SavepointState_Persisted) {
        scheduleDeleteNonPersistentSavepoint({entry});
    }
    _selectedEntry.reset();
}

void AutosaveWindow::onLoadSavepoint(SavepointEntry const& entry)
{
    FileTransferController::get().onOpenSimulation(entry->filename);
}

void AutosaveWindow::processCleanup()
{
    if (_scheduleCleanup) {
        printOverlayMessage("Cleaning up save points ...");

        auto nonPersistentEntries = SavepointTableService::get().truncate(_savepointTable.value(), 0);
        scheduleDeleteNonPersistentSavepoint(nonPersistentEntries);
        _scheduleCleanup = false;
    }
}

void AutosaveWindow::processAutomaticSavepoints()
{
    if (!_autosaveEnabled) {
        return;
    }

    auto minSinceLastAutosave = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - _lastAutosaveTimepoint).count();
    if (minSinceLastAutosave >= _autosaveInterval) {
        onCreateSavepoint(_catchPeaks != CatchPeaks_None);
        _lastAutosaveTimepoint = std::chrono::steady_clock::now();
    }

    if (_catchPeaks != CatchPeaks_None) {
        auto minSinceLastCatchPeak = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - _lastPeakTimepoint).count();
        if (minSinceLastCatchPeak >= PeakDetectionInterval) {
            _peakProcessor->executeTask(
                [&](auto const& senderId) {
                    return _persisterFacade->scheduleGetPeakSimulation(
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

void AutosaveWindow::scheduleDeleteNonPersistentSavepoint(std::vector<SavepointEntry> const& entries)
{
    for (auto const& entry : entries) {
        if (!entry->requestId.empty() && (entry->state == SavepointState_InQueue || entry->state == SavepointState_InProgress)) {
            _savepointsInProgressToDelete.emplace_back(entry);
        } else {

            //no request id => savepoint has been created on an other session
            SerializerService::get().deleteSimulation(entry->filename);
        }
    }
}

void AutosaveWindow::processDeleteNonPersistentSavepoint()
{
    std::vector<SavepointEntry> newRequestsToDelete;
    for (auto const& entry : _savepointsInProgressToDelete) {
        if (auto requestState = _persisterFacade->getRequestState(PersisterRequestId{entry->requestId})) {
            if (requestState.value() == PersisterRequestState::Finished || requestState.value() == PersisterRequestState::Error) {
                auto resultData = _persisterFacade->fetchSaveSimulationData(PersisterRequestId{entry->requestId});
                SerializerService::get().deleteSimulation(resultData.filename);
            } else {
                newRequestsToDelete.emplace_back(entry);
            }
        }
    }
    _savepointsInProgressToDelete = newRequestsToDelete;
}

void AutosaveWindow::scheduleCleanup()
{
    _scheduleCleanup = true;
}

void AutosaveWindow::updateSavepoint(int row)
{
    auto entry = _savepointTable->at(row);
    if (entry->state != SavepointState_Persisted) {
        auto newEntry = _savepointTable->at(row);
        auto requestState = _persisterFacade->getRequestState(PersisterRequestId{newEntry->requestId});
        if (requestState.has_value()) {
            if (requestState.value() == PersisterRequestState::InProgress) {
                newEntry->state = SavepointState_InProgress;
            }
            if (requestState.value() == PersisterRequestState::Finished) {
                newEntry->state = SavepointState_Persisted;
                auto requestResult = _persisterFacade->fetchPersisterRequestResult(PersisterRequestId{newEntry->requestId});

                if (auto saveResult = std::dynamic_pointer_cast<_SaveSimulationRequestResult>(requestResult)) {
                    auto const& data = saveResult->getData();
                    newEntry->timestep = data.timestep;
                    newEntry->timestamp = StringHelper::format(data.timestamp);
                    newEntry->name = data.projectName;
                    newEntry->filename = data.filename;
                } else if (auto saveResult = std::dynamic_pointer_cast<_SaveDeserializedSimulationRequestResult>(requestResult)) {
                    auto const& data = saveResult->getData();
                    newEntry->timestep = data.timestep;
                    newEntry->timestamp = StringHelper::format(data.timestamp);
                    newEntry->name = data.projectName;
                    newEntry->filename = data.filename;
                    newEntry->peak = StringHelper::format(toFloat(sumColorVector(data.rawStatisticsData.timeline.timestep.genomeComplexityVariance)), 2);
                    newEntry->peakType = "genome complexity variance";
                }
            }
            if (requestState.value() == PersisterRequestState::Error) {
                newEntry->state = SavepointState_Error;
            }
            SavepointTableService::get().updateEntry(_savepointTable.value(), row, newEntry);
        }
    }
}

void AutosaveWindow::updateSavepointTableFromFile()
{
    if (auto savepoint = SavepointTableService::get().loadFromFile(getSavepointFilename()); std::holds_alternative<SavepointTable>(savepoint)) {
        _savepointTable = std::get<SavepointTable>(savepoint);
    } else {
        _savepointTable.reset();
    }
    _selectedEntry.reset();
}

std::string AutosaveWindow::getSavepointFilename() const
{
    return (std::filesystem::path(_directory) / Const::SavepointTableFilename).string();
}

void AutosaveWindow::validationAndCorrection()
{
    _numberOfFiles = std::max(1, _numberOfFiles);
    _autosaveInterval = std::max(1, _autosaveInterval);
}
