#include "AutosaveDialog.h"

#include <algorithm>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include "AlienGui.h"
#include "LocalLibraryController.h"
#include "StyleService.h"

namespace
{
    auto constexpr DialogSize = RealVector2D{470.0f, 285.0f};
    auto constexpr FooterHeight = 50.0f;
    auto constexpr SettingsTextWidth = 110.0f;  // Fits the longest label of the settings

    void processWrappedText(std::string const& text, ImColor const& color)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, color.Value);
        ImGui::PushTextWrapPos(0.0f);
        AlienGui::Text(text);
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    }
}

AutosaveDialog::AutosaveDialog()
    : AlienDialog("Activate autosave", DialogSize)
{}

void AutosaveDialog::openIntern()
{
    auto const& controller = LocalLibraryController::get();
    _interval = controller.getAutosaveInterval();
    _folderName = controller.getAutosaveFolderName();
    _saveMode = controller.getSaveMode();
    _numberOfFiles = controller.getNumberOfFiles();
}

void AutosaveDialog::processIntern()
{
    auto folder = LocalLibraryController::get().resolveAutosaveFolder(_folderName);

    if (ImGui::BeginChild("##settings", {0, -scale(FooterHeight)})) {
        processSettings(folder.has_value());
        if (folder.has_value()) {
            ImGui::Spacing();
            processSummary(*folder);
        }
    }
    ImGui::EndChild();

    AlienGui::Separator();

    ImGui::BeginDisabled(!folder.has_value());
    if (AlienGui::ActionButton(
            AlienGui::ActionButtonParameters().buttonText("Activate").smallIcon(ICON_FA_PLAY).highlighted(true).frame(true).transparentBackground(false))) {
        onActivate();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void AutosaveDialog::processSettings(bool subfolderValid)
{
    AlienGui::InputInt(AlienGui::InputIntParameters().name("Interval (min)").textWidth(SettingsTextWidth), _interval);
    _interval = std::max(1, _interval);

    ImGui::BeginGroup();
    AlienGui::InputText(AlienGui::InputTextParameters().name("Subfolder").textWidth(SettingsTextWidth), _folderName);
    ImGui::EndGroup();
    AlienGui::Tooltip(AlienGui::TooltipParameters().text(
        "The subfolder in which the autosaves are stored, for instance one for each simulation run. Use / or \\ for nested subfolders. Without a "
        "subfolder, they are stored in the folder itself."));
    if (!subfolderValid) {
        processWrappedText("The subfolder is not valid. A name must not contain < > : \" | ? * and must not begin or end with a dot.", Const::WarningColor);
    }

    AlienGui::Switcher(
        AlienGui::SwitcherParameters().name("Mode").values({"Limited save files", "Unlimited save files"}).textWidth(SettingsTextWidth), &_saveMode);

    ImGui::BeginDisabled(_saveMode == LocalLibraryController::SaveMode_Unlimited);
    AlienGui::InputInt(AlienGui::InputIntParameters().name("Number of files").textWidth(SettingsTextWidth), _numberOfFiles);
    _numberOfFiles = std::max(1, _numberOfFiles);
    ImGui::EndDisabled();
}

void AutosaveDialog::processSummary(std::filesystem::path const& folder)
{
    auto preferredFolder = folder;
    preferredFolder.make_preferred();

    auto text = "The simulation is saved every " + (_interval == 1 ? std::string("minute") : std::to_string(_interval) + " minutes") + " into "
        + StringHelper::toUtf8(preferredFolder) + ".";
    if (_saveMode == LocalLibraryController::SaveMode_Limited) {
        text += _numberOfFiles == 1 ? " Only the newest autosave of this folder is kept."
                                    : " Only the " + std::to_string(_numberOfFiles) + " newest autosaves of this folder are kept.";
    }
    processWrappedText(text, Const::TextDimColor);
}

void AutosaveDialog::onActivate()
{
    auto& controller = LocalLibraryController::get();
    if (!controller.setAutosaveFolderName(_folderName)) {
        return;
    }
    controller.setAutosaveInterval(_interval);
    controller.setSaveMode(_saveMode);
    controller.setNumberOfFiles(_numberOfFiles);
    controller.setAutosaveEnabled(true);
    close();
}
