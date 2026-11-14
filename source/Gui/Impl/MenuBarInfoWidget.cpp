#include "MenuBarInfoWidget.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Data/SimulationParameters.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>

#include "AlienGui.h"
#include "BrowserController.h"
#include "BrowserData.h"
#include "BrowserWindow.h"
#include "NewSimulationService.h"
#include "StyleService.h"

namespace
{
    auto constexpr MenuButtonRounding = 7.0f;
    auto constexpr MenuButtonBorderSize = 2.0f;
    auto constexpr TooltipValueWidth = 120.0f;
}

MenuBarInfoWidget _MenuBarInfoWidget::create()
{
    return MenuBarInfoWidget(new _MenuBarInfoWidget());
}

namespace
{
    std::string getDisplayName(NetworkResourceRawTO const& rawTO)
    {
        auto projectName = std::string(_SimulationFacade::get()->getSimulationParameters().projectName.value);
        if (!projectName.empty() && projectName != SimulationParameters().projectName.value) {
            return projectName;
        }
        return NetworkResourceService::get().removeFoldersFromName(rawTO->resourceName);
    }

    std::string getNameText(std::string const& displayName)
    {
        return " " + displayName;
    }

    std::string getUserText(NetworkResourceRawTO const& rawTO)
    {
        return "  by " + rawTO->userName + " ";
    }

    std::string getHeartText(NetworkResourceRawTO const& rawTO)
    {
        return " " ICON_FA_HEART " " + std::to_string(rawTO->getTotalLikes()) + " ";
    }

    void pushMenuButtonStyle()
    {
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, MenuButtonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, MenuButtonBorderSize);
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::MenuButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::MenuButtonActiveColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::MenuButtonHoveredColor);
    }

    void popMenuButtonStyle()
    {
        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(2);
    }
}

void _MenuBarInfoWidget::process()
{
    if (!AlienGui::IsMenuBarVisible()) {
        return;
    }
    processCurrentSimulation();
}

void _MenuBarInfoWidget::processCurrentSimulation()
{
    if (BrowserWindow::get().isOn()) {
        return;
    }
    auto const& originResourceId = NewSimulationService::get().getOriginResourceId();
    if (!originResourceId.has_value()) {
        return;
    }
    auto rawTO = BrowserController::get().getData()->findRawTO(*originResourceId);
    if (rawTO == nullptr) {
        return;
    }

    auto displayName = getDisplayName(rawTO);
    auto const& style = ImGui::GetStyle();
    auto nameWidth = ImGui::CalcTextSize(getNameText(displayName).c_str()).x + ImGui::CalcTextSize(getUserText(rawTO).c_str()).x + style.FramePadding.x * 2;
    auto heartWidth = ImGui::CalcTextSize(getHeartText(rawTO).c_str()).x + style.FramePadding.x * 2;
    auto totalWidth = nameWidth + style.ItemSpacing.x + heartWidth;

    auto cursorX = ImGui::GetCursorPosX();
    auto startX = cursorX + ImGui::GetContentRegionAvail().x - totalWidth;
    if (startX < cursorX) {
        return;
    }
    ImGui::SetCursorPosX(startX);

    processNameButton(rawTO, displayName, nameWidth);
    processHeartButton(rawTO, displayName);
}

namespace
{
    void processTooltipRow(std::string const& label, std::string const& value)
    {
        AlienGui::Text(value);
        ImGui::SameLine(scale(TooltipValueWidth));
        AlienGui::Text(AlienGui::TextParameters().text(label).style(AlienGui::TextStyle::Decent));
    }
}

void _MenuBarInfoWidget::processNameButton(NetworkResourceRawTO const& rawTO, std::string const& displayName, float width)
{
    auto pos = ImGui::GetCursorScreenPos();
    pushMenuButtonStyle();
    auto clicked = ImGui::Button("##currentSimulationName", {width, 0});
    popMenuButtonStyle();

    auto const& style = ImGui::GetStyle();
    auto nameText = getNameText(displayName);
    auto textPos = ImVec2{pos.x + style.FramePadding.x, pos.y + style.FramePadding.y};
    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), nameText.c_str());
    drawList->AddText(
        {textPos.x + ImGui::CalcTextSize(nameText.c_str()).x, textPos.y}, ImGui::GetColorU32(Const::TextDimColor.Value), getUserText(rawTO).c_str());

    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextTooltipColor.Value);
        AlienGui::Text(AlienGui::TextParameters().text(displayName).style(AlienGui::TextStyle::Bold));
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserLeafTextColor);
        AlienGui::Text(rawTO->resourceName);
        ImGui::PopStyleColor();
        AlienGui::Separator();
        processTooltipRow("User", rawTO->userName);
        processTooltipRow("Uploaded", rawTO->timestamp.substr(0, 10));
        processTooltipRow("Reactions", std::to_string(rawTO->getTotalLikes()));
        processTooltipRow("Downloads", std::to_string(rawTO->numDownloads));
        AlienGui::Separator();
        AlienGui::Text("Click to show it in the browser.");
        ImGui::PopStyleColor();
        ImGui::EndTooltip();
    }
    if (clicked) {
        BrowserWindow::get().showResource(rawTO->resourceType, rawTO->workspaceType, rawTO->id);
    }
}

void _MenuBarInfoWidget::processHeartButton(NetworkResourceRawTO const& rawTO, std::string const& displayName)
{
    auto const& data = BrowserController::get().getData();
    auto isOwnReaction = data->getOwnEmojiType(rawTO->id).has_value();
    pushMenuButtonStyle();
    ImGui::PushStyleColor(ImGuiCol_Text, isOwnReaction ? Const::BrowserOwnReactionButtonTextColor.Value : Const::TextDimColor.Value);
    auto clicked = ImGui::Button((getHeartText(rawTO) + "###currentSimulationHeart").c_str());
    ImGui::PopStyleColor();
    popMenuButtonStyle();

    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextTooltipColor.Value);
        if (isOwnReaction) {
            AlienGui::Text(AlienGui::TextParameters().text("You reacted to " + displayName).style(AlienGui::TextStyle::Bold));
            AlienGui::Text("Click to take your reaction back.");
        } else {
            AlienGui::Text(AlienGui::TextParameters().text("Give " + rawTO->userName + " a heart").style(AlienGui::TextStyle::Bold));
            if (!NetworkService::get().isLoggedIn()) {
                AlienGui::Text("No account needed.");
            }
            AlienGui::Text("It is also kept in the Liked list of the browser.");
        }
        ImGui::PopStyleColor();
        ImGui::EndTooltip();
    }
    if (clicked) {
        data->onToggleHeart(rawTO);
    }
}
