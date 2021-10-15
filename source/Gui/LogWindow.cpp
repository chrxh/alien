#include "LogWindow.h"

#include "imgui.h"

#include <boost/range/adaptor/reversed.hpp>

#include "StyleRepository.h"
#include "SimpleLogger.h"

_LogWindow::_LogWindow(StyleRepository const& styleRepository, SimpleLogger const& logger)
    : _styleRepository(styleRepository)
    , _logger(logger)
{}

void _LogWindow::process()
{
    if (!_on) {
        return;
    }
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
    if (ImGui::Begin("Log", &_on)) {
        ImGui::Checkbox("Verbose", &_verbose);

/*
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
*/ 
        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::BeginChild("##", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::PushFont(_styleRepository->getMonospaceFont());
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::LogMessageColor);

        for (auto const& logMessage :
             _logger->getMessages(_verbose ? Priority::Unimportant : Priority::Important) | boost::adaptors::reversed) {
            ImGui::Text(logMessage.c_str());
        }
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::EndChild();

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
