#include "AlienWindow.h"

#include <imgui.h>

#include "GlobalSettings.h"
#include "StyleRepository.h"

_AlienWindow::_AlienWindow(std::string const& title, std::string const& settingsNode, bool defaultOn)
    : _title(title)
    , _settingsNode(settingsNode)
{
    _on = GlobalSettings::getInstance().getBoolState(settingsNode + ".active", defaultOn);
}

_AlienWindow::~_AlienWindow()
{
    GlobalSettings::getInstance().setBoolState(_settingsNode + ".active", _on);
}

void _AlienWindow::process()
{
    processBackground();

    if (!_on) {
        return;
    }
    ImGui::PushID(_title.c_str());

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin(_title.c_str(), &_on)) {
        processIntern();
    }
    ImGui::End();

    ImGui::PopID();
}

bool _AlienWindow::isOn() const
{
    return _on;
}

void _AlienWindow::setOn(bool value)
{
    _on = value;
    if (value) {
        processActivated();
    }
}
