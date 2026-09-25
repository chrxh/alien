#include "McpWindow.h"

#include <algorithm>
#include <ranges>
#include <format>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include "AlienGui.h"
#include "McpController.h"
#include "McpSettingsDialog.h"
#include "OverlayController.h"
#include "StyleService.h"

namespace
{
    auto constexpr BadgePadding = 10.0f;
    auto constexpr BadgeDotRadius = 4.0f;
    auto constexpr BadgeDotSpacing = 8.0f;
    auto constexpr StepNumberRadius = 9.0f;
    auto constexpr StepNumberSpacing = 8.0f;
    auto constexpr TimeColumnSample = "0000-00-00 00:00:00";
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
        processConnectionGuide();
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
             })),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_COG).name("Settings").action([&] { McpSettingsDialog::get().open(); }))});
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

void McpWindow::processConnectionGuide()
{
    auto& controller = McpController::get();
    auto const& style = ImGui::GetStyle();

    ImGui::Spacing();
    AlienGui::Text(AlienGui::TextParameters().text("Connect your AI agent").style(AlienGui::TextStyle::Bold));

    ImGui::SameLine();
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
                "Works with any AI agent or MCP client that supports MCP servers of type HTTP (also called Streamable HTTP), regardless of the AI "
                "provider.\n\nOnly agents on this computer can connect.\n\nAvailable tools: {}",
                toolNames);
        },
        false);

    processStepNumber(1);
    ImGui::TextUnformatted("In your AI agent, add an MCP server of type HTTP.");

    processStepNumber(2);
    ImGui::TextUnformatted("Use this URL:");

    auto url = controller.getServerUrl();
    auto copyButtonWidth = ImGui::CalcTextSize(CopyButtonText).x + style.FramePadding.x * 2;
    ImGui::SameLine();
    ImGui::PushFont(StyleService::get().getMonospaceMediumFont());
    ImGui::PushStyleColor(ImGuiCol_FrameBg, Const::BackgroundColor.Value);
    ImGui::PushStyleColor(ImGuiCol_Border, Const::LineColor.Value);
    ImGui::PushStyleColor(ImGuiCol_Text, Const::SoftHighlightTextColor.Value);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    auto urlWidth = ImGui::CalcTextSize(url.c_str()).x + style.FramePadding.x * 2;
    ImGui::SetNextItemWidth(std::min(urlWidth, ImGui::GetContentRegionAvail().x - copyButtonWidth - style.ItemSpacing.x));
    ImGui::InputText("##url", url.data(), url.size() + 1, ImGuiInputTextFlags_ReadOnly);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);
    ImGui::PopFont();

    ImGui::SameLine();
    if (ImGui::Button(CopyButtonText)) {
        ImGui::SetClipboardText(url.c_str());
        printOverlayMessage("Copied to clipboard");
    }
}

void McpWindow::processStepNumber(int number)
{
    ImGui::AlignTextToFramePadding();
    auto radius = scale(StepNumberRadius);
    auto pos = ImGui::GetCursorScreenPos();
    ImVec2 center{pos.x + radius, pos.y + ImGui::GetFrameHeight() / 2};
    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddCircleFilled(center, radius, Const::RaisedColor);

    auto text = std::to_string(number);
    ImGui::PushFont(StyleService::get().getSmallBoldFont());
    auto textSize = ImGui::CalcTextSize(text.c_str());
    drawList->AddText({center.x - textSize.x / 2, center.y - textSize.y / 2}, Const::AccentColor, text.c_str());
    ImGui::PopFont();

    ImGui::Dummy({radius * 2, ImGui::GetFrameHeight()});
    ImGui::SameLine(0, scale(StepNumberSpacing));
}

void McpWindow::processCommandLog()
{
    auto const& commandLog = McpController::get().getCommandLog();

    ImGui::Spacing();
    AlienGui::Group(AlienGui::GroupParameters().text(std::format("Command log ({})", commandLog.size())));

    auto flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("##mcpCommands", 3, flags, ImVec2(-1, -1))) {
        auto timeColumnWidth = ImGui::CalcTextSize(TimeColumnSample).x + ImGui::GetStyle().CellPadding.x * 2;
        ImGui::TableSetupColumn("Timestamp", ImGuiTableColumnFlags_WidthFixed, timeColumnWidth);
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
                AlienGui::Text(StringHelper::format(entry.time));

                ImGui::TableNextColumn();
                ImGui::PushFont(StyleService::get().getMonospaceMediumFont());
                ImGui::PushStyleColor(ImGuiCol_Text, Const::SoftHighlightTextColor.Value);
                AlienGui::Text(AlienGui::TextParameters().text(entry.command).truncate(true));
                ImGui::PopStyleColor();
                ImGui::PopFont();
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
