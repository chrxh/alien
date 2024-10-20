#include "AutosaveWindow.h"

#include <Fonts/IconsFontAwesome5.h>

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"

#include "AlienImGui.h"
#include "OverlayMessageController.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto constexpr RightColumnWidth = 200.0f;
    auto constexpr AutosaveSenderId = "Autosave";
}

void AutosaveWindow::init(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade)
{
    _simulationFacade = simulationFacade;
    _persisterFacade = persisterFacade;
    _settingsOpen = GlobalSettings::get().getBool("windows.autosave.settings.open", _settingsOpen);
    _settingsHeight = GlobalSettings::get().getFloat("windows.autosave.settings.height", _settingsHeight);
    _autosaveEnabled = GlobalSettings::get().getBool("windows.autosave.enabled", _autosaveEnabled);
    _origAutosaveInterval = GlobalSettings::get().getInt("windows.autosave.interval", _origAutosaveInterval);
    _autosaveInterval = _origAutosaveInterval;
    _origSaveMode = GlobalSettings::get().getInt("windows.autosave.mode", _origSaveMode);
    _saveMode = _origSaveMode;
    _numberOfFiles = GlobalSettings::get().getInt("windows.autosave.number of files", _origNumberOfFiles);
}

AutosaveWindow::AutosaveWindow()
    : AlienWindow("Autosave (work in progress)", "windows.autosave", false)
{}

void AutosaveWindow::shutdownIntern()
{
    GlobalSettings::get().setBool("windows.autosave.settings.open", _settingsOpen);
    GlobalSettings::get().setFloat("windows.autosave.settings.height", _settingsHeight);
    GlobalSettings::get().setBool("windows.autosave.enabled", _autosaveEnabled);
    GlobalSettings::get().setInt("windows.autosave.interval", _autosaveInterval);
    GlobalSettings::get().setInt("windows.autosave.mode", _saveMode);
    GlobalSettings::get().setInt("windows.autosave.number of files", _numberOfFiles);
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
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
        createSavepoint();
    }
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
        clipper.Begin(_savePoints.size());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto& entry = _savePoints[row];
                updateSavepoint(entry);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(15.0f));

                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(row + 1));

                ImGui::TableNextColumn();
                if (entry.state == SavepointState::InQueue) {
                    AlienImGui::Text("In queue");
                }
                if (entry.state == SavepointState::InProgress) {
                    AlienImGui::Text("In progress");
                }
                if (entry.state == SavepointState::Persisted) {
                    AlienImGui::Text(entry.timestamp);
                }
                if (entry.state == SavepointState::Error) {
                    AlienImGui::Text("Error");
                }

                ImGui::TableNextColumn();
                if (entry.state == SavepointState::Persisted) {
                    AlienImGui::Text(entry.name);
                }

                ImGui::TableNextColumn();
                if (entry.state == SavepointState::Persisted) {
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
            AlienImGui::InputText(AlienImGui::InputTextParameters().name("Directory").textWidth(RightColumnWidth).defaultValue(_origLocation), _location);
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

    _savePoints.emplace_front(SavepointState::InQueue, jobId.value, "", "", 0);
}

void AutosaveWindow::updateSavepoint(SavepointEntry& savepoint)
{
    if (savepoint.state != SavepointState::Persisted) {
        auto requestState = _persisterFacade->getRequestState(PersisterRequestId(savepoint.id));
        if (requestState == PersisterRequestState::InProgress) {
            savepoint.state = SavepointState::InProgress;
        }
        if (requestState == PersisterRequestState::Finished) {
            savepoint.state = SavepointState::Persisted;
            auto jobResult = _persisterFacade->fetchSavedSimulationData(PersisterRequestId{savepoint.id});
            savepoint.timestep = jobResult.timestep;
            savepoint.timestamp = StringHelper::format(jobResult.timestamp);
            savepoint.name = jobResult.name;
        }
        if (requestState == PersisterRequestState::Error) {
            savepoint.state = SavepointState::Error;
        }
    }
}

void AutosaveWindow::validationAndCorrection()
{
    _numberOfFiles = std::max(1, _numberOfFiles);
}
