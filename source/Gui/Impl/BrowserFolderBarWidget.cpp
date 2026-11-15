#include "BrowserFolderBarWidget.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <cctype>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>
#include <Base/WebLinkHelper.h>

#include "AlienGui.h"
#include "AutosaveDialog.h"
#include "BrowserData.h"
#include "GenericFileDialog.h"
#include "LocalLibraryController.h"
#include "StyleService.h"

namespace
{
    auto constexpr FolderFieldWidth = 266.0f;
    auto constexpr MinFolderFieldWidth = 150.0f;
    auto constexpr FolderButtonWidth = 30.0f;
    auto constexpr AutosaveMarkerRadius = 4.0f;

    auto constexpr FolderLabel = "Folder";

    // The two buttons and the label behind the field of the folder
    float calcFolderControlsWidth()
    {
        auto const& style = ImGui::GetStyle();
        auto buttonsWidth = (style.ItemSpacing.x - style.FramePadding.x + scale(FolderButtonWidth)) * 2;
        return buttonsWidth + style.ItemSpacing.x + ImGui::CalcTextSize(FolderLabel).x;
    }

    AlienGui::ActionButtonParameters getActivateButton()
    {
        return AlienGui::ActionButtonParameters()
            .buttonText("Activate autosave")
            .smallIcon(ICON_FA_PLAY)
            .highlighted(true)
            .frame(true)
            .transparentBackground(false)
            .tooltip("Save the simulation automatically at an interval.");
    }

    AlienGui::ActionButtonParameters getDeactivateButton()
    {
        return AlienGui::ActionButtonParameters().buttonText("Deactivate autosave").smallIcon(ICON_FA_STOP).transparentBackground(false);
    }

    // Digits are measured as zeros, so a countdown does not change what fits into the row
    float calcStableTextWidth(std::string text)
    {
        std::ranges::replace_if(text, [](unsigned char character) { return std::isdigit(character) != 0; }, '0');
        return ImGui::CalcTextSize(text.c_str()).x;
    }

    std::string getAutosaveInfo(std::chrono::seconds durationUntilNextAutosave, bool showInterval)
    {
        auto countdown = StringHelper::format(durationUntilNextAutosave);
        if (showInterval) {
            return "Every " + std::to_string(LocalLibraryController::get().getAutosaveInterval()) + " min, next in " + countdown;
        }
        return "Next in " + countdown;
    }
}

BrowserFolderBarWidget _BrowserFolderBarWidget::create(BrowserData const& data)
{
    return BrowserFolderBarWidget(new _BrowserFolderBarWidget(data));
}

_BrowserFolderBarWidget::_BrowserFolderBarWidget(BrowserData const& data)
    : _data(data)
{}

float _BrowserFolderBarWidget::processFolder(float maxWidth)
{
    auto& controller = LocalLibraryController::get();
    auto const& style = ImGui::GetStyle();
    auto isAutosave = _data->getCurrentWorkspaceId().workspaceType == WorkspaceType_Autosave;
    auto folder = isAutosave ? controller.getAutosaveMainFolder() : controller.getLibraryFolder();
    auto folderText = StringHelper::toUtf8(folder);
    auto controlsWidth = calcFolderControlsWidth();
    auto fieldWidth = std::clamp(maxWidth - controlsWidth, scale(MinFolderFieldWidth), scale(FolderFieldWidth));

    // The end of a long path is shown, because it tells the folders apart
    auto pos = ImGui::GetCursorScreenPos();
    auto size = ImVec2{fieldWidth, ImGui::GetFrameHeight()};
    auto textWidth = ImGui::CalcTextSize(folderText.c_str()).x;
    auto innerWidth = size.x - style.FramePadding.x * 2;
    ImGui::Dummy(size);
    if (textWidth > innerWidth) {
        AlienGui::Tooltip(AlienGui::TooltipParameters().text(folderText).delay(false));
    }
    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, {pos.x + size.x, pos.y + size.y}, ImGui::GetColorU32(ImGuiCol_FrameBg), style.FrameRounding);
    drawList->PushClipRect({pos.x + style.FramePadding.x, pos.y}, {pos.x + size.x - style.FramePadding.x, pos.y + size.y}, true);
    drawList->AddText(
        {pos.x + style.FramePadding.x + std::min(0.0f, innerWidth - textWidth), pos.y + style.FramePadding.y}, Const::TextBaseColor, folderText.c_str());
    drawList->PopClipRect();

    // The autosaves in progress keep their folder
    auto isFolderFixed = isAutosave && controller.isAutosaveEnabled();
    ImGui::SameLine();
    AlienGui::MoveTickLeft();
    ImGui::BeginDisabled(isFolderFixed);
    if (AlienGui::Button(ICON_FA_FOLDER_OPEN, FolderButtonWidth)) {
        GenericFileDialog::get().showOpenFileDialog("Select directory", "", folderText, [isAutosave](std::filesystem::path const& path) {
            if (isAutosave) {
                LocalLibraryController::get().setAutosaveMainFolder(path);
            } else {
                LocalLibraryController::get().setLibraryFolder(path);
            }
        });
    }
    ImGui::EndDisabled();
    AlienGui::Tooltip(AlienGui::TooltipParameters().text(
        isFolderFixed    ? "Deactivate autosave to choose another folder."
            : isAutosave ? "Choose another folder for the autosaves. The files stay where they are."
                         : "Choose another folder for My simulations and My genomes. The files stay where they are."));

    ImGui::SameLine();
    AlienGui::MoveTickLeft();
    if (AlienGui::Button(ICON_FA_EXTERNAL_LINK_ALT, FolderButtonWidth)) {
        std::error_code error;
        std::filesystem::create_directories(folder, error);
        WebLinkHelper::openInFileManager(folder);
    }
    AlienGui::Tooltip(AlienGui::TooltipParameters().text("Show the folder in the file manager."));

    ImGui::SameLine();
    AlienGui::Text(FolderLabel);

    return fieldWidth + controlsWidth;
}

float _BrowserFolderBarWidget::getMinFolderWidth() const
{
    return scale(MinFolderFieldWidth) + calcFolderControlsWidth();
}

void _BrowserFolderBarWidget::processAutosave()
{
    auto& controller = LocalLibraryController::get();
    auto availableWidth = ImGui::GetContentRegionAvail().x;

    if (!controller.isAutosaveEnabled()) {
        if (AlienGui::ActionButton(getActivateButton())) {
            AutosaveDialog::get().open();
        }
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
        AlienGui::Text("Autosave is off");
        ImGui::PopStyleColor();
        return;
    }

    if (AlienGui::ActionButton(getDeactivateButton())) {
        controller.setAutosaveEnabled(false);
    }

    // The interval is left out where the row is too narrow for it
    auto frameHeight = ImGui::GetFrameHeight();
    auto markerRadius = scale(AutosaveMarkerRadius);
    auto infoOffset = AlienGui::CalcActionButtonWidth(getDeactivateButton()) + ImGui::GetStyle().ItemSpacing.x * 2 + markerRadius * 2;
    auto longestDuration = std::chrono::minutes(controller.getAutosaveInterval());
    auto showInterval = infoOffset + calcStableTextWidth(getAutosaveInfo(longestDuration, true)) <= availableWidth;

    ImGui::SameLine();
    auto markerPos = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddCircleFilled({markerPos.x + markerRadius, markerPos.y + frameHeight / 2}, markerRadius, Const::AccentColor);
    ImGui::Dummy({markerRadius * 2, frameHeight});
    ImGui::SameLine();
    AlienGui::Text(getAutosaveInfo(controller.getDurationUntilNextAutosave(), showInterval));
}
