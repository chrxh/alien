#include "LogWindow.h"

#include "imgui.h"

#include <boost/range/adaptor/reversed.hpp>

#include "StyleRepository.h"
#include "GuiLogger.h"

_LogWindow::_LogWindow(StyleRepository const& styleRepository, GuiLogger const& logger)
    : _styleRepository(styleRepository)
    , _logger(logger)
{}

void _LogWindow::process()
{
    if (!_on) {
        return;
    }
    if (ImGui::Begin("Log", &_on)) {
        ImGui::Checkbox("Verbose", &_verbose);

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::PushFont(_styleRepository->getMonospaceFont());
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::LogMessageColor);

        for (auto const& logMessage :
             _logger->getMessages(_verbose ? Priority::Unimportant : Priority::Important) | boost::adaptors::reversed) {
            ImGui::Text(logMessage.c_str());
        }
        ImGui::PopStyleColor();
        ImGui::PopFont();

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
