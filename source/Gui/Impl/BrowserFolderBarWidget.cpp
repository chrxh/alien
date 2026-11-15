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
    auto constexpr SeparatorSpace = 20.0f;
    auto constexpr SeparatorMargin = 3.0f;
    auto constexpr AutosaveMarkerRadius = 4.0f;

    auto constexpr FolderLabel = "Folder";
    auto constexpr AutosaveOffText = "Autosave is off";

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

    // Digits are measured as zeros, so a countdown does not move what surrounds it
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

    // The same whether the autosave is on or off, so its button stays where it is
    float calcAutosaveWidth(bool showInfo, bool showInterval)
    {
        auto const& style = ImGui::GetStyle();
        auto activeWidth = AlienGui::CalcActionButtonWidth(getDeactivateButton());
        auto inactiveWidth = AlienGui::CalcActionButtonWidth(getActivateButton());
        if (showInfo) {
            auto longestDuration = std::chrono::minutes(LocalLibraryController::get().getAutosaveInterval());
            activeWidth += style.ItemSpacing.x * 2 + scale(AutosaveMarkerRadius) * 2 + calcStableTextWidth(getAutosaveInfo(longestDuration, showInterval));
            inactiveWidth += style.ItemSpacing.x + ImGui::CalcTextSize(AutosaveOffText).x;
        }
        return std::max(activeWidth, inactiveWidth);
    }
}

BrowserFolderBarWidget _BrowserFolderBarWidget::create(BrowserData const& data)
{
    return BrowserFolderBarWidget(new _BrowserFolderBarWidget(data));
}

_BrowserFolderBarWidget::_BrowserFolderBarWidget(BrowserData const& data)
    : _data(data)
{}

void _BrowserFolderBarWidget::process()
{
    auto workspaceId = _data->getCurrentWorkspaceId();
    auto const& style = ImGui::GetStyle();
    auto availableWidth = ImGui::GetContentRegionAvail().x;

    auto buttonsWidth = (style.ItemSpacing.x - style.FramePadding.x + scale(FolderButtonWidth)) * 2;
    auto labelWidth = style.ItemSpacing.x + ImGui::CalcTextSize(FolderLabel).x;
    auto fieldWidth = availableWidth - buttonsWidth - labelWidth;

    // My simulations leaves the room of the autosave free, so the folder looks the same in both workspaces. The less room there is, the less is
    // told next to the button of the autosave.
    auto showInfo = false;
    auto showInterval = false;
    auto autosaveWidth = 0.0f;
    if (workspaceId.resourceType == NetworkResourceType_Simulation) {
        auto fits = [&](bool info, bool interval) {
            return fieldWidth - scale(SeparatorSpace) - calcAutosaveWidth(info, interval) >= scale(MinFolderFieldWidth);
        };
        showInterval = fits(true, true);
        showInfo = showInterval || fits(true, false);
        autosaveWidth = calcAutosaveWidth(showInfo, showInterval);
        fieldWidth -= scale(SeparatorSpace) + autosaveWidth;
    }
    fieldWidth = std::clamp(fieldWidth, scale(MinFolderFieldWidth), scale(FolderFieldWidth));

    auto startPosX = ImGui::GetCursorPosX();
    processFolder(fieldWidth);

    if (workspaceId.workspaceType == WorkspaceType_Autosave) {
        auto folderEndPosX = startPosX + fieldWidth + buttonsWidth + labelWidth;
        ImGui::SameLine(std::max(folderEndPosX + scale(SeparatorSpace), startPosX + availableWidth - autosaveWidth));
        processAutosave(showInfo, showInterval);
    }
}

void _BrowserFolderBarWidget::processFolder(float fieldWidth)
{
    auto& controller = LocalLibraryController::get();
    auto const& style = ImGui::GetStyle();
    auto isAutosave = _data->getCurrentWorkspaceId().workspaceType == WorkspaceType_Autosave;
    auto folder = isAutosave ? controller.getAutosaveMainFolder() : controller.getLibraryFolder();
    auto folderText = StringHelper::toUtf8(folder);

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
}

void _BrowserFolderBarWidget::processAutosave(bool showInfo, bool showInterval)
{
    auto& controller = LocalLibraryController::get();
    auto frameHeight = ImGui::GetFrameHeight();
    auto drawList = ImGui::GetWindowDrawList();

    auto pos = ImGui::GetCursorScreenPos();
    auto separatorPosX = pos.x - scale(SeparatorSpace) / 2;
    drawList->AddLine(
        {separatorPosX, pos.y + scale(SeparatorMargin)}, {separatorPosX, pos.y + frameHeight - scale(SeparatorMargin)}, ImGui::GetColorU32(ImGuiCol_Border));

    if (!controller.isAutosaveEnabled()) {
        if (AlienGui::ActionButton(getActivateButton())) {
            AutosaveDialog::get().open();
        }
        if (showInfo) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
            AlienGui::Text(AutosaveOffText);
            ImGui::PopStyleColor();
        }
        return;
    }

    if (AlienGui::ActionButton(getDeactivateButton())) {
        controller.setAutosaveEnabled(false);
    }
    if (showInfo) {
        ImGui::SameLine();
        auto markerPos = ImGui::GetCursorScreenPos();
        auto markerRadius = scale(AutosaveMarkerRadius);
        drawList->AddCircleFilled({markerPos.x + markerRadius, markerPos.y + frameHeight / 2}, markerRadius, Const::AccentColor);
        ImGui::Dummy({markerRadius * 2, frameHeight});
        ImGui::SameLine();
        AlienGui::Text(getAutosaveInfo(controller.getDurationUntilNextAutosave(), showInterval));
    }
}
