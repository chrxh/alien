#include "LogWindow.h"

#include "imgui.h"

#include <boost/range/adaptor/reversed.hpp>

#include "StyleRepository.h"
#include "SimpleLogger.h"
#include "GlobalSettings.h"

_LogWindow::_LogWindow(StyleRepository const& styleRepository, SimpleLogger const& logger)
    : _styleRepository(styleRepository)
    , _logger(logger)
{
    _on = GlobalSettings::getInstance().getBoolState("windows.log.active", false);
    _verbose = GlobalSettings::getInstance().getBoolState("windows.log.verbose", false);
}

_LogWindow::~_LogWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.log.active", _on);
    GlobalSettings::getInstance().setBoolState("windows.log.verbose", _verbose);
}

void _LogWindow::process()
{
    if (!_on) {
        return;
    }
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin("Log", &_on)) {

        if (ImGui::BeginChild(
                "##", ImVec2(0, ImGui::GetContentRegionAvail().y - 40), true, ImGuiWindowFlags_HorizontalScrollbar)) {
            ImGui::PushFont(_styleRepository->getMonospaceFont());
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::LogMessageColor);

            for (auto const& logMessage : _logger->getMessages(_verbose ? Priority::Unimportant : Priority::Important)
                     | boost::adaptors::reversed) {
                ImGui::Text(logMessage.c_str());
            }
            ImGui::PopStyleColor();
            ImGui::PopFont();
            ImGui::EndChild();
        }
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Checkbox("Verbose", &_verbose);

        ImGui::End();
    }
}

bool _LogWindow::isOn() const
{
    return _on;
}

void _LogWindow::setOn(bool value)
{
    _on = value;
}
