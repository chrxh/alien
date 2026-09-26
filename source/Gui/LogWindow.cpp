#include "LogWindow.h"

#include <ranges>

#include <imgui.h>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include "AlienGui.h"
#include "GuiLogger.h"
#include "StyleService.h"

namespace
{
    auto constexpr TimeColumnSample = "0000-00-00 00:00:00";
    auto constexpr ToolbarHeight = 40.0f;
}

void LogWindow::initIntern()
{
    _logger = std::make_shared<_GuiLogger>();
    _verbose = GlobalSettings::get().getValue("windows.log.verbose", false);
}

LogWindow::LogWindow()
    : AlienWindow("Log", "windows.log", false, false, {934.0f, 636.0f}, {652.0f, 414.0f})
{}

void LogWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.log.verbose", _verbose);
}

void LogWindow::processIntern()
{
    auto const& messages = _logger->getMessages(_verbose ? Priority::Unimportant : Priority::Important);

    auto flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("##logMessages", 2, flags, ImVec2(-1, ImGui::GetContentRegionAvail().y - scale(ToolbarHeight)))) {
        auto timeColumnWidth = ImGui::CalcTextSize(TimeColumnSample).x + ImGui::GetStyle().CellPadding.x * 2;
        ImGui::TableSetupColumn("Timestamp", ImGuiTableColumnFlags_WidthFixed, timeColumnWidth);
        ImGui::TableSetupColumn("Message", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        ImGuiListClipper clipper;
        clipper.Begin(toInt(messages.size()));
        while (clipper.Step()) {
            for (auto row : std::views::iota(clipper.DisplayStart, clipper.DisplayEnd)) {
                auto const& message = messages.at(messages.size() - 1 - row);
                ImGui::PushID(row);
                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                AlienGui::Text(StringHelper::format(message.time));

                ImGui::TableNextColumn();
                auto firstLine = message.text.substr(0, message.text.find('\n'));
                if (firstLine.size() < message.text.size()) {
                    firstLine += " ...";
                }
                auto textColor = message.priority == Priority::Important ? Const::AccentColor : Const::TextDimColor;
                ImGui::PushStyleColor(ImGuiCol_Text, textColor.Value);
                AlienGui::Text(AlienGui::TextParameters().text(firstLine).truncate(true));
                ImGui::PopStyleColor();
                AlienGui::Tooltip(message.text);

                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::Spacing();
    AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Verbose"), _verbose);
}
