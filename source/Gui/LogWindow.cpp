#include "LogWindow.h"

#include <imgui.h>

#include <boost/range/adaptor/reversed.hpp>

#include "StyleRepository.h"
#include "SimpleLogger.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"

_LogWindow::_LogWindow(SimpleLogger const& logger)
    : _AlienWindow("Log", "windows.log", false)
    , _logger(logger)
{
    _verbose = GlobalSettings::getInstance().getBoolState("windows.log.verbose", false);
}

_LogWindow::~_LogWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.log.verbose", _verbose);
}

void _LogWindow::processIntern()
{
    auto styleRepository = StyleRepository::getInstance();
    if (ImGui::BeginChild(
            "##", ImVec2(0, ImGui::GetContentRegionAvail().y - styleRepository.scaleContent(40.0f)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::LogMessageColor);

        for (auto const& logMessage : _logger->getMessages(_verbose ? Priority::Unimportant : Priority::Important) | boost::adaptors::reversed) {
            ImGui::TextUnformatted(logMessage.c_str());
        }
        ImGui::PopStyleColor();
        ImGui::PopFont();
    }
    ImGui::EndChild();

    ImGui::Spacing();
    ImGui::Spacing();
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Verbose"), _verbose);
}
