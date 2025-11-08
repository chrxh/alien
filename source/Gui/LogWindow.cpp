#include "LogWindow.h"

#include <boost/range/adaptor/reversed.hpp>

#include <imgui.h>

#include <Base/GlobalSettings.h>

#include "AlienGui.h"
#include "GuiLogger.h"
#include "StyleRepository.h"

void LogWindow::initIntern()
{
    _logger = std::make_shared<_GuiLogger>();
    _verbose = GlobalSettings::get().getValue("windows.log.verbose", false);
}

LogWindow::LogWindow()
    : AlienWindow("Log", "windows.log", false)
{}

void LogWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.log.verbose", _verbose);
}

void LogWindow::processIntern()
{
    auto& styleRepository = StyleRepository::get();
    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - styleRepository.scale(40.0f)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
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
    AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Verbose"), _verbose);
}
