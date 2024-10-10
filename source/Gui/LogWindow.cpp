#include "LogWindow.h"

#include <imgui.h>

#include <boost/range/adaptor/reversed.hpp>

#include "Base/GlobalSettings.h"

#include "StyleRepository.h"
#include "GuiLogger.h"
#include "AlienImGui.h"

_LogWindow::_LogWindow(GuiLogger const& logger)
    : _AlienWindow("Log", "windows.log", false)
    , _logger(logger)
{
    _verbose = GlobalSettings::get().getBool("windows.log.verbose", false);
}

_LogWindow::~_LogWindow()
{
    GlobalSettings::get().setBool("windows.log.verbose", _verbose);
}

void _LogWindow::processIntern()
{
    auto& styleRepository = StyleRepository::get();
    if (ImGui::BeginChild(
            "##", ImVec2(0, ImGui::GetContentRegionAvail().y - styleRepository.scale(40.0f)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGui::PushFont(StyleRepository::get().getMonospaceMediumFont());
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::MonospaceColor);

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
