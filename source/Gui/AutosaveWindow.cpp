#include "AutosaveWindow.h"

#include <filesystem>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

#include <PersisterInterface/SavepointTableService.h>

#include "AlienGui.h"
#include "AutosaveController.h"
#include "FileTransferController.h"
#include "GenericMessageDialog.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr RightColumnWidth = 200.0f;
    auto constexpr DefaultSettingsHeight = 130.0f;
}

AutosaveWindow::AutosaveWindow()
    : AlienWindow("Autosave", "windows.autosave", false, true, {49.0f, 71.0f}, {546.0f, 448.0f})
{}

void AutosaveWindow::initIntern()
{
    _settingsOpen = GlobalSettings::get().getValue("windows.autosave.settings.open", _settingsOpen);
    _settingsHeight =
        GlobalSettings::get().getValue("windows.autosave.settings.height", scale(DefaultSettingsHeight)) * WindowController::get().getContentScaleCorrection();

    _origAutosaveInterval = GlobalSettings::get().getValue("windows.autosave.interval", _origAutosaveInterval);
    _origSaveMode = GlobalSettings::get().getValue("windows.autosave.mode", _origSaveMode);
    _origNumberOfFiles = GlobalSettings::get().getValue("windows.autosave.number of files", _origNumberOfFiles);
    _origDirectory = GlobalSettings::get().getValue("windows.autosave.directory", (std::filesystem::current_path() / Const::AutosavePath).string());
}

void AutosaveWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.autosave.settings.open", _settingsOpen);
    GlobalSettings::get().setValue("windows.autosave.settings.height", _settingsHeight);
}

void AutosaveWindow::processIntern()
{
    try {
        processToolbar();

        if (ImGui::BeginChild("##child1", {0, -scale(44.0f)})) {
            processHeader();

            //AlienGui::Separator();
            if (ImGui::BeginChild("##child2", {0, _settingsOpen ? -_settingsHeight : -scale(35.0f)})) {
                processTable();
            }
            ImGui::EndChild();

            processSettings();
        }
        ImGui::EndChild();

        processStatusBar();
    } catch (std::runtime_error const& error) {
        GenericMessageDialog::get().information("Error", error.what());
    }
}

void AutosaveWindow::processToolbar()
{
    auto const& savepointTable = AutosaveController::get().getSavepointTable();

    AlienGui::Toolbar(
        AlienGui::ToolbarParameters().id("Autosave"),
        {AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_PLUS).name("Create save point").disabled(!savepointTable.has_value()).action([&] {
                 AutosaveController::get().onCreateSavepoint(false);
             })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_MINUS).name("Delete save point").disabled(!static_cast<bool>(_selectedEntry)).action([&] {
                 AutosaveController::get().onDeleteSavepoint(_selectedEntry);
                 _selectedEntry.reset();
             })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_BROOM)
                                                 .name("Delete all save points")
                                                 .disabled(!savepointTable.has_value() || savepointTable->isEmpty())
                                                 .action([&] {
                                                     GenericMessageDialog::get().yesNo("Delete", "Do you really want to delete all savepoints?", [&]() {
                                                         AutosaveController::get().scheduleCleanup();
                                                         _selectedEntry.reset();
                                                     });
                                                 }))});
}

void AutosaveWindow::processHeader() {}

void AutosaveWindow::processTable()
{
    auto const& savepointTable = AutosaveController::get().getSavepointTable();
    if (!savepointTable.has_value()) {
        AlienGui::Text("Error: Savepoint files could not be read or created in the specified directory.");
        return;
    }
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Save files", 3, flags, ImVec2(-1, -1), 0.0f)) {
        ImGui::TableSetupColumn("Simulation", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Timestamp", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Time step", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
        //ImGui::TableSetupColumn("Peak value", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(200.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        ImGuiListClipper clipper;
        clipper.Begin(savepointTable->getSize());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto const& entry = savepointTable->at(row);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(23.0f));

                // Project name
                ImGui::TableNextColumn();
                if (entry->state == SavepointState_InQueue) {
                    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
                    AlienGui::Text("In queue");
                    ImGui::PopStyleColor();
                } else if (entry->state == SavepointState_InProgress) {
                    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
                    AlienGui::Text("In progress");
                    ImGui::PopStyleColor();
                } else if (entry->state == SavepointState_Persisted) {
                    auto triggerLoadSavepoint = AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_DOWNLOAD));
                    AlienGui::Tooltip("Load save point", false);
                    if (triggerLoadSavepoint) {
                        onLoadSavepoint(entry);
                    }

                    ImGui::SameLine();
                    AlienGui::Text(entry->name);
                } else if (entry->state == SavepointState_Error) {
                    AlienGui::Text("Error");
                }
                ImGui::SameLine();
                ImGui::Dummy({0, scale(22.0f)});

                ImGui::SameLine();
                auto selected = _selectedEntry == entry;
                if (AlienGui::TableRowSelectable(
                        "",
                        &selected,
                        ImGuiSelectableFlags_None,
                        RealVector2D(0, scale(ImGui::GetTextLineHeightWithSpacing()) - ImGui::GetStyle().FramePadding.y))) {
                    _selectedEntry = selected ? entry : nullptr;
                }

                // Timestamp
                ImGui::TableNextColumn();
                if (entry->state == SavepointState_Persisted) {
                    AlienGui::Text(entry->timestamp);
                }

                // Timestep
                ImGui::TableNextColumn();
                if (entry->state == SavepointState_Persisted) {
                    AlienGui::Text(AlienGui::TextParameters().text(StringHelper::format(entry->timestep)).rightAligned(true));
                }

                // Peak
                //ImGui::TableNextColumn();
                //AlienGui::Text(entry->peak);

                //if (!entry->peakType.empty()) {
                //    ImGui::SameLine();
                //    AlienGui::Text(AlienGui::TextParameters().text(" (" + entry->peakType + ")").style(AlienGui::TextStyle::Decent));
                //}

                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
}

void AutosaveWindow::processSettings()
{
    ImGui::Spacing();
    ImGui::Spacing();
    if (_settingsOpen) {
        AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _settingsHeight);
    }

    _settingsOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Settings").rank(AlienGui::TreeNodeRank::High).defaultOpen(_settingsOpen));
    if (_settingsOpen) {
        if (ImGui::BeginChild("##autosaveSettings", {scale(0), 0})) {
            auto autosaveEnabled = AutosaveController::get().isAutosaveEnabled();
            auto autosaveInterval = AutosaveController::get().getAutosaveInterval();
            if (AlienGui::InputInt(
                    AlienGui::InputIntParameters().name("Autosave interval (min)").textWidth(RightColumnWidth).defaultValue(_origAutosaveInterval),
                    autosaveInterval,
                    &autosaveEnabled)) {
                AutosaveController::get().setAutosaveInterval(autosaveInterval);
                AutosaveController::get().setAutosaveEnabled(autosaveEnabled);
            }
            //if (AlienGui::Switcher(
            //        AlienGui::SwitcherParameters()
            //            .name("Catch peaks")
            //            .textWidth(RightColumnWidth)
            //            .defaultValue(_origCatchPeaks)
            //            .readOnly(!_autosaveEnabled)
            //            .values({
            //                "None",
            //                "Genome complexity variance",
            //            })
            //            .tooltip("If activated, the simulation is monitored continuously. When the autosave interval expires, the time at which the selected "
            //                     "measured value was particularly high is saved."),
            //        _catchPeaks)) {
            //    _peakDeserializedSimulation->setDeserializedSimulation(SimulationDesc());
            //}

            auto directory = AutosaveController::get().getDirectory();
            if (AlienGui::InputText(
                    AlienGui::InputTextParameters()
                        .name("Directory")
                        .textWidth(RightColumnWidth)
                        .defaultValue(_origDirectory)
                        .folderButton(true)
                        .tooltip("The directory where the savepoints are stored can be chosen here. This allows the savepoints to be created in a separate "
                                 "directory for a simulation run. The savepoints are named using the current time step."),
                    directory)) {
                AutosaveController::get().setDirectory(directory);
                _selectedEntry.reset();
            }

            auto saveMode = AutosaveController::get().getSaveMode();
            if (AlienGui::Switcher(
                    AlienGui::SwitcherParameters()
                        .name("Mode")
                        .values({"Limited save files", "Unlimited save files"})
                        .textWidth(RightColumnWidth)
                        .defaultValue(_origSaveMode),
                    &saveMode)) {
                AutosaveController::get().setSaveMode(saveMode);
            }

            if (saveMode == AutosaveController::SaveMode_Circular) {
                auto numberOfFiles = AutosaveController::get().getNumberOfFiles();
                if (AlienGui::InputInt(
                        AlienGui::InputIntParameters().name("Number of files").textWidth(RightColumnWidth).defaultValue(_origNumberOfFiles), numberOfFiles)) {
                    AutosaveController::get().setNumberOfFiles(numberOfFiles);
                }
            }
        }
        ImGui::EndChild();
    }
    AlienGui::EndTreeNode();
}

void AutosaveWindow::processStatusBar()
{
    auto const& savepointTable = AutosaveController::get().getSavepointTable();

    std::vector<std::string> statusItems;
    if (!savepointTable.has_value()) {
        statusItems.emplace_back("No valid directory");
    } else if (!AutosaveController::get().isAutosaveEnabled()) {
        statusItems.emplace_back("No autosave scheduled");
    } else {
        statusItems.emplace_back("Next autosave in " + StringHelper::format(AutosaveController::get().getDurationUntilNextAutosave()));
    }
    if (savepointTable.has_value()) {
        statusItems.emplace_back(std::to_string(savepointTable->getSize()) + " save points");
    }

    AlienGui::StatusBar(statusItems);
}

void AutosaveWindow::onLoadSavepoint(SavepointEntry const& entry)
{
    auto path = SavepointTableService::get().calcAbsolutePath(AutosaveController::get().getSavepointTable().value(), entry);
    FileTransferController::get().onOpenSimulation(path);
}
