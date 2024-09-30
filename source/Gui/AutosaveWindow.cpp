#include "AutosaveWindow.h"

#include <Fonts/IconsFontAwesome5.h>

#include "Base/GlobalSettings.h"

#include "AlienImGui.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto constexpr RightColumnWidth = 200.0f;
}

_AutosaveWindow::_AutosaveWindow(SimulationController const& simController)
    : _AlienWindow("Autosave", "windows.autosave", false)
    , _simController(simController)
    , _worker(simController)
{
    _settingsOpen = GlobalSettings::getInstance().getBool("windows.autosave.settings.open", _settingsOpen);
    _settingsHeight = GlobalSettings::getInstance().getFloat("windows.autosave.settings.height", _settingsHeight);
    _autosaveEnabled = GlobalSettings::getInstance().getBool("windows.autosave.enabled", _autosaveEnabled);
    _origAutosaveInterval = GlobalSettings::getInstance().getInt("windows.autosave.interval", _origAutosaveInterval);
    _autosaveInterval = _origAutosaveInterval;
    _origSaveMode = GlobalSettings::getInstance().getInt("windows.autosave.mode", _origSaveMode);
    _saveMode = _origSaveMode;
    _numberOfFiles = GlobalSettings::getInstance().getInt("windows.autosave.number of files", _origNumberOfFiles);

    _thread = new std::thread(&PersisterWorker::runThreadLoop, &_worker);
}

_AutosaveWindow::~_AutosaveWindow()
{
    _worker.shutdown();
    _thread->join();
    delete _thread;

    GlobalSettings::getInstance().setBool("windows.autosave.settings.open", _settingsOpen);
    GlobalSettings::getInstance().setFloat("windows.autosave.settings.height", _settingsHeight);
    GlobalSettings::getInstance().setBool("windows.autosave.enabled", _autosaveEnabled);
    GlobalSettings::getInstance().setInt("windows.autosave.interval", _autosaveInterval);
    GlobalSettings::getInstance().setInt("windows.autosave.mode", _saveMode);
    GlobalSettings::getInstance().setInt("windows.autosave.number of files", _numberOfFiles);
}

void _AutosaveWindow::processIntern()
{
    processToolbar();

    processHeader();

    AlienImGui::Separator();
    if (ImGui::BeginChild("", {0, _settingsOpen ? -scale(_settingsHeight) : -scale(50.0f)})) {
        processTable();
    }
    ImGui::EndChild();

    processSettings();

    validationAndCorrection();
}

void _AutosaveWindow::processToolbar()
{
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
        onCreateSave();
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

void _AutosaveWindow::processHeader()
{
    AlienImGui::InputInt(
        AlienImGui::InputIntParameters().name("Autosave interval (min)").textWidth(RightColumnWidth).defaultValue(_origAutosaveInterval),
        _autosaveInterval,
        &_autosaveEnabled);
}

void _AutosaveWindow::processTable()
{
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Save files", 4, flags, ImVec2(0, 0), 0.0f)) {
        ImGui::TableSetupColumn("No", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(30.0f));
        ImGui::TableSetupColumn("Timestamp", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(90.0f));
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(200.0f));
        ImGui::TableSetupColumn("Time step", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(_savePoints.size());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto item = &_savePoints[row];

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(15.0f));

                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->sequenceNumber));

                ImGui::TableNextColumn();
                AlienImGui::Text(item->timestamp);

                ImGui::TableNextColumn();
                AlienImGui::Text(item->name);

                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->timestep));

                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
}

void _AutosaveWindow::processSettings()
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
            AlienImGui::InputText(AlienImGui::InputTextParameters().name("Location on disc").textWidth(RightColumnWidth).defaultValue(_origLocation), _location);
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

void _AutosaveWindow::onCreateSave()
{
    DeserializedSimulation dummy;
    _worker.saveToDisc("d:\\test.sim", Viewport::getZoomFactor(), Viewport::getCenterInWorldPos());
}

void _AutosaveWindow::validationAndCorrection()
{
    _numberOfFiles = std::max(1, _numberOfFiles);
}
