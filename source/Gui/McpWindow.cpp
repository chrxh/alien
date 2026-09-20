#include "McpWindow.h"

#include <ranges>
#include <format>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include "AlienGui.h"
#include "McpController.h"
#include "OverlayController.h"
#include "StyleService.h"

namespace
{
    auto constexpr BadgePadding = 10.0f;
    auto constexpr BadgeDotRadius = 4.0f;
    auto constexpr BadgeDotSpacing = 8.0f;
    auto constexpr CardRounding = 6.0f;
    auto constexpr CardPaddingX = 10.0f;
    auto constexpr CardPaddingY = 8.0f;
    auto constexpr StatusColumnWidth = 20.0f;
    auto constexpr TimeColumnWidth = 70.0f;
    auto constexpr ResultColumnWeight = 1.4f;
    auto constexpr CopyButtonText = ICON_FA_COPY "  Copy";
}

McpWindow::McpWindow()
    : AlienWindow("MCP server", "windows.mcp server", false, false, {60.0f, 60.0f}, {620.0f, 700.0f})
{}

void McpWindow::processIntern()
{
    processToolbar();

    if (ImGui::BeginChild("##content", {0, 0})) {
        processStatusBadge();
        processConnectionCard();
        processCommandLog();
    }
    ImGui::EndChild();
}

void McpWindow::processToolbar()
{
    auto& controller = McpController::get();
    auto running = controller.isServerRunning();
    AlienGui::Toolbar(
        AlienGui::ToolbarParameters().id("McpServer"),
        {AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_PLAY).name("Start server").disabled(running).action([&] { controller.setServerRunning(true); })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_STOP).name("Stop server").disabled(!running).action([&] { controller.setServerRunning(false); })),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_BROOM).name("Clear command log").disabled(controller.getCommandLog().empty()).action([&] {
                 controller.clearCommandLog();
             }))});
}

void McpWindow::processStatusBadge()
{
    auto running = McpController::get().isServerRunning();
    auto text = running ? "Running" : "Stopped";
    auto textSize = ImGui::CalcTextSize(text);
    auto paddingX = scale(BadgePadding);
    auto paddingY = ImGui::GetStyle().FramePadding.y;
    auto dotRadius = scale(BadgeDotRadius);
    auto width = paddingX * 2 + dotRadius * 2 + scale(BadgeDotSpacing) + textSize.x;
    auto height = textSize.y + paddingY * 2;

    auto pos = ImGui::GetCursorScreenPos();
    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, {pos.x + width, pos.y + height}, running ? Const::McpRunningBadgeColor : Const::RaisedColor, height / 2);
    drawList->AddCircleFilled({pos.x + paddingX + dotRadius, pos.y + height / 2}, dotRadius, running ? Const::McpSuccessColor : Const::TextFaintColor);
    drawList->AddText(
        {pos.x + paddingX + dotRadius * 2 + scale(BadgeDotSpacing), pos.y + paddingY}, running ? Const::McpSuccessColor : Const::TextDimColor, text);
    ImGui::Dummy({width, height});
}

void McpWindow::processConnectionCard()
{
    auto& controller = McpController::get();
    auto const& style = ImGui::GetStyle();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, Const::PanelColor.Value);
    ImGui::PushStyleColor(ImGuiCol_Border, Const::LineColor.Value);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, scale(CardRounding));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {scale(CardPaddingX), scale(CardPaddingY)});
    if (ImGui::BeginChild("##connection", {0, 0}, ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
        auto rightEdge = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
        AlienGui::Text(AlienGui::TextParameters().text("Connect an MCP client").style(AlienGui::TextStyle::Bold));

        ImGui::SameLine(rightEdge - ImGui::CalcTextSize(ICON_FA_QUESTION_CIRCLE).x);
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextInfoColor.Value);
        ImGui::TextUnformatted(ICON_FA_QUESTION_CIRCLE);
        ImGui::PopStyleColor();
        AlienGui::Tooltip(
            [&] {
                std::string toolNames;
                for (auto const& toolName : controller.getToolNames()) {
                    toolNames += (toolNames.empty() ? "" : ", ") + toolName;
                }
                return std::format(
                    "Works with any MCP client that supports the HTTP transport (also called Streamable HTTP), regardless of the AI provider.\n\n"
                    "Only clients on this computer can connect.\n\nAvailable tools: {}",
                    toolNames);
            },
            false);

        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
        ImGui::TextUnformatted("Add a server of type HTTP with this URL in your MCP client.");
        ImGui::PopStyleColor();

        auto url = controller.getServerUrl();
        auto copyButtonWidth = ImGui::CalcTextSize(CopyButtonText).x + style.FramePadding.x * 2;
        ImGui::PushFont(StyleService::get().getMonospaceMediumFont());
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - copyButtonWidth - style.ItemSpacing.x);
        ImGui::InputText("##url", url.data(), url.size() + 1, ImGuiInputTextFlags_ReadOnly);
        ImGui::PopFont();

        ImGui::SameLine();
        if (ImGui::Button(CopyButtonText)) {
            ImGui::SetClipboardText(url.c_str());
            printOverlayMessage("Copied to clipboard");
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

void McpWindow::processCommandLog()
{
    auto const& commandLog = McpController::get().getCommandLog();

    ImGui::Spacing();
    AlienGui::Group(AlienGui::GroupParameters().text(std::format("Command log ({})", commandLog.size())));

    auto flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("##commandLog", 4, flags, ImVec2(-1, -1))) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize, scale(StatusColumnWidth));
        ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, scale(TimeColumnWidth));
        ImGui::TableSetupColumn("Command", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Result", ImGuiTableColumnFlags_WidthStretch, ResultColumnWeight);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        ImGuiListClipper clipper;
        clipper.Begin(toInt(commandLog.size()));
        while (clipper.Step()) {
            for (auto row : std::views::iota(clipper.DisplayStart, clipper.DisplayEnd)) {
                auto const& entry = commandLog.at(commandLog.size() - 1 - row);
                ImGui::PushID(row);
                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                ImGui::PushStyleColor(ImGuiCol_Text, entry.isError ? Const::WarningColor.Value : Const::McpSuccessColor.Value);
                ImGui::TextUnformatted(entry.isError ? ICON_FA_TIMES : ICON_FA_CHECK);
                ImGui::PopStyleColor();

                ImGui::TableNextColumn();
                AlienGui::Text(StringHelper::formatTimeOfDay(entry.time));
                AlienGui::Tooltip(StringHelper::format(entry.time));

                ImGui::TableNextColumn();
                AlienGui::Text(AlienGui::TextParameters().text(entry.command).style(AlienGui::TextStyle::Monospace).truncate(true));
                AlienGui::Tooltip(entry.command);

                ImGui::TableNextColumn();
                if (entry.isError) {
                    ImGui::PushStyleColor(ImGuiCol_Text, Const::WarningColor.Value);
                }
                AlienGui::Text(AlienGui::TextParameters().text(entry.result).truncate(true));
                if (entry.isError) {
                    ImGui::PopStyleColor();
                }
                AlienGui::Tooltip(entry.result);

                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
}
