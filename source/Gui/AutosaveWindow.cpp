#include "AutosaveWindow.h"

#include <filesystem>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "PersisterInterface/SavepointTableService.h"

#include "AlienImGui.h"
#include "OverlayController.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto constexpr RightColumnWidth = 200.0f;
    auto constexpr AutosaveSenderId = "Autosave";
}

AutosaveWindow::AutosaveWindow()
    : AlienWindow("Autosave (work in progress)", "windows.autosave", false)
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
    _numberOfFiles = GlobalSettings::get().getValue("windows.autosave.number of files", _origNumberOfFiles);
    _origDirectory = GlobalSettings::get().getValue("windows.autosave.directory", std::filesystem::current_path().string());
    _directory = _origDirectory;
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
}

void AutosaveWindow::processIntern()
{
    processToolbar();

    processHeader();

    AlienImGui::Separator();
    if (ImGui::BeginChild("##autosave", {0, _settingsOpen ? -scale(_settingsHeight) : -scale(50.0f)})) {
        processTable();
    }
    ImGui::EndChild();

    processSettings();

    validationAndCorrection();
}

void AutosaveWindow::processToolbar()
{
    ImGui::SameLine();
    ImGui::BeginDisabled(!_savepointTable.has_value());
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
        createSavepoint();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Create save point");

    ImGui::SameLine();
    ImGui::BeginDisabled(true);
    if (AlienImGui::ToolbarButton(ICON_FA_TRASH)) {
    }
    AlienImGui::Tooltip("Delete save point");
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(true);
    if (AlienImGui::ToolbarButton(ICON_FA_BROOM)) {
    }
    AlienImGui::Tooltip("Delete all save points");
    ImGui::EndDisabled();

    AlienImGui::Separator();
}

void AutosaveWindow::processHeader()
{
    AlienImGui::InputInt(
        AlienImGui::InputIntParameters().name("Autosave interval (min)").textWidth(RightColumnWidth).defaultValue(_origAutosaveInterval),
        _autosaveInterval,
        &_autosaveEnabled);
}

void AutosaveWindow::processTable()
{
    if (!_savepointTable.has_value()) {
        return;
    }
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Save files", 4, flags, ImVec2(0, 0), 0.0f)) {
        ImGui::TableSetupColumn("No", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(30.0f));
        ImGui::TableSetupColumn("Timestamp", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(200.0f));
        ImGui::TableSetupColumn("Time step", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(_savepointTable->getSize());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                updateSavepoint(row);
                auto const& entry = _savepointTable->at(row);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(15.0f));

                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(row + 1));

                ImGui::TableNextColumn();
                if (entry.state == SavepointState_InQueue) {
                    AlienImGui::Text("In queue");
                }
                if (entry.state == SavepointState_InProgress) {
                    AlienImGui::Text("In progress");
                }
                if (entry.state == SavepointState_Persisted) {
                    AlienImGui::Text(entry.timestamp);
                }
                if (entry.state == SavepointState_Error) {
                    AlienImGui::Text("Error");
                }

                ImGui::TableNextColumn();
                if (entry.state == SavepointState_Persisted) {
                    AlienImGui::Text(entry.name);
                }

                ImGui::TableNextColumn();
                if (entry.state == SavepointState_Persisted) {
                    AlienImGui::Text(StringHelper::format(entry.timestep));
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
            AlienImGui::InputText(AlienImGui::InputTextParameters().name("Directory").textWidth(RightColumnWidth).defaultValue(_origDirectory), _directory);
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Mode")
                    .values({"Circular save files", "Unlimited save files"})
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

void AutosaveWindow::createSavepoint()
{
    printOverlayMessage("Creating save point ...");
    static int i = 0;
    auto senderInfo = SenderInfo{.senderId = SenderId{AutosaveSenderId}, .wishResultData = true, .wishErrorInfo = true};
    auto saveData = SaveSimulationRequestData{"d:\\test" + std::to_string(++i) + ".sim", Viewport::get().getZoomFactor(), Viewport::get().getCenterInWorldPos()};
    auto jobId = _persisterFacade->scheduleSaveSimulationToFile(senderInfo, saveData);

    SavepointTableService::get().insertEntry(
        _savepointTable.value(), SavepointEntry{.id = jobId.value, .filename = "", .state = SavepointState_InQueue, .timestamp = "", .name = "", .timestep = 0});
}

void AutosaveWindow::updateSavepoint(int row)
{
    auto entry = _savepointTable->at(row);
    if (entry.state != SavepointState_Persisted) {
        auto newEntry = _savepointTable->at(row);
        auto requestState = _persisterFacade->getRequestState(PersisterRequestId{newEntry.id});
        if (requestState.has_value()) {
            if (requestState.value() == PersisterRequestState::InProgress) {
                newEntry.state = SavepointState_InProgress;
            }
            if (requestState.value() == PersisterRequestState::Finished) {
                newEntry.state = SavepointState_Persisted;
                auto jobResult = _persisterFacade->fetchSaveSimulationData(PersisterRequestId{newEntry.id});
                newEntry.timestep = jobResult.timestep;
                newEntry.timestamp = StringHelper::format(jobResult.timestamp);
                newEntry.name = jobResult.name;
            }
            if (requestState.value() == PersisterRequestState::Error) {
                newEntry.state = SavepointState_Error;
            }
            SavepointTableService::get().updateEntry(_savepointTable.value(), row, newEntry);
        }
    }
}

void AutosaveWindow::validationAndCorrection()
{
    _numberOfFiles = std::max(1, _numberOfFiles);
}
