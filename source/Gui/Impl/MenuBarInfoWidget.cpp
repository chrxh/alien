#include "MenuBarInfoWidget.h"

#include <algorithm>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <Data/SimulationParameters.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>

#include "AlienGui.h"
#include "BrowserController.h"
#include "BrowserData.h"
#include "BrowserWindow.h"
#include "LocalLibraryController.h"
#include "NewSimulationService.h"
#include "StyleService.h"

namespace
{
    auto constexpr MenuButtonRounding = 7.0f;
    auto constexpr MenuButtonBorderSize = 2.0f;
    auto constexpr TooltipValueWidth = 120.0f;
    auto constexpr LocalTooltipValueWidth = 150.0f;     // Leaves room for a date with time
    auto constexpr TimestampLengthWithoutSeconds = 16;  // e.g. 2026-11-13 14:02
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

    // Unlike the entries of the browser, the content of the library is also kept up to date while the browser is closed
    LibraryFile const* findLocalFile()
    {
        auto const& localFile = NewSimulationService::get().getLocalFile();
        if (!localFile.has_value()) {
            return nullptr;
        }
        auto const& simulations = LocalLibraryController::get().getContent().simulations;
        auto findResult = std::ranges::find_if(simulations, [&](LibraryFile const& simulation) { return simulation.path == *localFile; });
        return findResult != simulations.end() ? &*findResult : nullptr;
    }

    bool alignRightIfFits(float width)
    {
        auto cursorX = ImGui::GetCursorPosX();
        auto startX = cursorX + ImGui::GetContentRegionAvail().x - width;
        if (startX < cursorX) {
            return false;
        }
        ImGui::SetCursorPosX(startX);
        return true;
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

    // A simulation of the server takes precedence. While the browser receives the entries of the server, at the start and at every refresh,
    // a simulation of this computer is only shown if it has been shown before.
    if (auto const& originResourceId = NewSimulationService::get().getOriginResourceId()) {
        if (auto rawTO = BrowserController::get().getData()->findRawTO(*originResourceId)) {
            _localSimulationShown = false;
            processServerSimulation(rawTO);
            return;
        }
        if (BrowserController::get().isRefreshing() && !_localSimulationShown) {
            return;
        }
    }
    auto file = findLocalFile();
    _localSimulationShown = file != nullptr;
    if (file != nullptr) {
        processLocalSimulation(*file);
    }
}

namespace
{
    // The byline follows the name in a dimmed color. Returns true if the button has been clicked.
    bool processNameButton(std::string const& nameText, std::string const& bylineText, float width)
    {
        auto pos = ImGui::GetCursorScreenPos();
        pushMenuButtonStyle();
        auto result = ImGui::Button("##currentSimulationName", {width, 0});
        popMenuButtonStyle();

        auto const& style = ImGui::GetStyle();
        auto textPos = ImVec2{pos.x + style.FramePadding.x, pos.y + style.FramePadding.y};
        auto drawList = ImGui::GetWindowDrawList();
        drawList->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), nameText.c_str());
        drawList->AddText({textPos.x + ImGui::CalcTextSize(nameText.c_str()).x, textPos.y}, ImGui::GetColorU32(Const::TextDimColor.Value), bylineText.c_str());
        return result;
    }
}

void _MenuBarInfoWidget::processServerSimulation(NetworkResourceRawTO const& rawTO)
{
    auto displayName = getDisplayName(rawTO);
    auto nameText = getNameText(displayName);
    auto userText = getUserText(rawTO);
    auto const& style = ImGui::GetStyle();
    auto nameWidth = ImGui::CalcTextSize(nameText.c_str()).x + ImGui::CalcTextSize(userText.c_str()).x + style.FramePadding.x * 2;
    auto heartWidth = ImGui::CalcTextSize(getHeartText(rawTO).c_str()).x + style.FramePadding.x * 2;
    if (!alignRightIfFits(nameWidth + style.ItemSpacing.x + heartWidth)) {
        return;
    }

    auto clicked = processNameButton(nameText, userText, nameWidth);
    if (ImGui::IsItemHovered()) {
        processServerTooltip(rawTO, displayName);
    }
    if (clicked) {
        BrowserWindow::get().showResource(rawTO->resourceType, rawTO->workspaceType, rawTO->id);
    }
    processHeartButton(rawTO, displayName);
}

void _MenuBarInfoWidget::processLocalSimulation(LibraryFile const& file)
{
    auto displayName = NetworkResourceService::get().removeFoldersFromName(file.name);
    auto nameText = getNameText(displayName) + " ";
    auto nameWidth = ImGui::CalcTextSize(nameText.c_str()).x + ImGui::GetStyle().FramePadding.x * 2;
    if (!alignRightIfFits(nameWidth)) {
        return;
    }

    auto clicked = processNameButton(nameText, std::string(), nameWidth);
    if (ImGui::IsItemHovered()) {
        processLocalTooltip(file, displayName);
    }
    if (clicked) {
        auto workspaceType = file.kind == LibraryFileKind::Named ? WorkspaceType_Local : WorkspaceType_Autosave;
        BrowserWindow::get().showResource(NetworkResourceType_Simulation, workspaceType, StringHelper::toUtf8(file.path));
    }
}

namespace
{
    void processTooltipRow(std::string const& label, std::string const& value, float valueWidth = TooltipValueWidth)
    {
        AlienGui::Text(value);
        ImGui::SameLine(scale(valueWidth));
        AlienGui::Text(AlienGui::TextParameters().text(label).style(AlienGui::TextStyle::Decent));
    }
}

void _MenuBarInfoWidget::processServerTooltip(NetworkResourceRawTO const& rawTO, std::string const& displayName)
{
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

void _MenuBarInfoWidget::processLocalTooltip(LibraryFile const& file, std::string const& displayName)
{
    ImGui::BeginTooltip();
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextTooltipColor.Value);
    AlienGui::Text(AlienGui::TextParameters().text(displayName).style(AlienGui::TextStyle::Bold));
    if (file.name != displayName) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserLeafTextColor);
        AlienGui::Text(file.name);
        ImGui::PopStyleColor();
    }
    AlienGui::Separator();
    auto timestamp = file.timestamp.substr(0, TimestampLengthWithoutSeconds);
    processTooltipRow(file.kind == LibraryFileKind::Autosave ? "Autosaved" : "Saved", timestamp, LocalTooltipValueWidth);
    processTooltipRow("File size", StringHelper::formatFileSize(file.fileSize), LocalTooltipValueWidth);
    AlienGui::Separator();
    AlienGui::Text("Click to show it in the browser.");
    ImGui::PopStyleColor();
    ImGui::EndTooltip();
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
