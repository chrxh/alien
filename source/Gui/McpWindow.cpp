#include "McpWindow.h"

#include <algorithm>
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
    auto constexpr RightColumnWidth = 120.0f;
    auto constexpr CopyButtonWidth = 35.0f;
    auto constexpr TimeColumnWidth = 140.0f;
    auto constexpr MinCommandLogHeight = 150.0f;
}

McpWindow::McpWindow()
    : AlienWindow("MCP server", "windows.mcp server", false, false, {60.0f, 60.0f}, {620.0f, 700.0f})
{}

void McpWindow::processIntern()
{
    processToolbar();

    if (ImGui::BeginChild("##content", {0, 0})) {
        processServerSettings();
        processConnectionInfo();
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

void McpWindow::processServerSettings()
{
    auto& controller = McpController::get();
    AlienGui::Group(AlienGui::GroupParameters().text("Server"));

    auto running = controller.isServerRunning();
    AlienGui::Text(running ? "Running at " + controller.getServerUrl() : "Stopped");

    ImGui::BeginDisabled(running);
    auto port = controller.getPort();
    if (AlienGui::InputInt(AlienGui::InputIntParameters().name("Port").textWidth(RightColumnWidth), port)) {
        controller.setPort(port);
    }
    ImGui::EndDisabled();
}

void McpWindow::processConnectionInfo()
{
    auto& controller = McpController::get();
    auto url = controller.getServerUrl();
    AlienGui::Group(AlienGui::GroupParameters().text("Connection"));

    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
    ImGui::TextWrapped(
        "%s",
        std::format(
            "ALIEN provides an MCP server with the Streamable HTTP transport. It only accepts connections from this computer. Available tools: {}.",
            [&] {
                std::string result;
                for (auto const& toolName : controller.getToolNames()) {
                    result += (result.empty() ? "" : ", ") + toolName;
                }
                return result;
            }())
            .c_str());
    ImGui::PopStyleColor();

    AlienGui::Text("Server URL");
    processCopyableText("url", url);

    AlienGui::Text("Claude Code");
    processCopyableText("claudeCode", "claude mcp add --transport http alien " + url);

    AlienGui::Text("Configuration file of MCP clients with HTTP support (e.g. .mcp.json)");
    processCopyableText(
        "httpConfig",
        std::format(
            "{{\n"
            "  \"mcpServers\": {{\n"
            "    \"alien\": {{\n"
            "      \"type\": \"http\",\n"
            "      \"url\": \"{}\"\n"
            "    }}\n"
            "  }}\n"
            "}}",
            url),
        9);

    AlienGui::Text("Configuration file of MCP clients with stdio support only (e.g. Claude Desktop), requires Node.js");
    processCopyableText(
        "stdioConfig",
        std::format(
            "{{\n"
            "  \"mcpServers\": {{\n"
            "    \"alien\": {{\n"
            "      \"command\": \"npx\",\n"
            "      \"args\": [\"mcp-remote\", \"{}\"]\n"
            "    }}\n"
            "  }}\n"
            "}}",
            url),
        9);
}

void McpWindow::processCommandLog()
{
    auto const& commandLog = McpController::get().getCommandLog();
    AlienGui::Group(AlienGui::GroupParameters().text("Command log"));

    auto height = std::max(ImGui::GetContentRegionAvail().y, scale(MinCommandLogHeight));
    auto flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("##commandLog", 3, flags, ImVec2(-1, height))) {
        ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, scale(TimeColumnWidth));
        ImGui::TableSetupColumn("Command", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Result", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        for (auto const& entry : commandLog | std::views::reverse) {
            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            AlienGui::Text(StringHelper::format(entry.time));

            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", entry.command.c_str());

            ImGui::TableNextColumn();
            if (entry.isError) {
                ImGui::PushStyleColor(ImGuiCol_Text, Const::DangerColor.Value);
            }
            ImGui::TextWrapped("%s", entry.result.c_str());
            if (entry.isError) {
                ImGui::PopStyleColor();
            }
        }
        ImGui::EndTable();
    }
}

void McpWindow::processCopyableText(std::string const& id, std::string text, int numLines)
{
    auto const& style = ImGui::GetStyle();
    ImGui::PushFont(StyleService::get().getMonospaceMediumFont());
    auto size = ImVec2(
        ImGui::GetContentRegionAvail().x - scale(CopyButtonWidth) - style.ItemSpacing.x,
        ImGui::GetTextLineHeight() * toFloat(numLines) + style.FramePadding.y * 2);
    ImGui::InputTextMultiline(("##" + id).c_str(), text.data(), text.size() + 1, size, ImGuiInputTextFlags_ReadOnly);
    ImGui::PopFont();

    ImGui::SameLine();
    if (AlienGui::Button(ICON_FA_COPY "##" + id, CopyButtonWidth)) {
        ImGui::SetClipboardText(text.c_str());
        printOverlayMessage("Copied to clipboard");
    }
    AlienGui::Tooltip("Copy to clipboard");
}
